import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model
import numpy as np

class StatisticsAlignmentLayer(nn.Module):
    """
    Learned-only statistical alignment layer.
    Expects `prompt_stats` with key 'stats': {batch_idx: {stat_name: value}}.
    """
    def __init__(self, pred_len, hidden_dim=64, dropout=0.1, enabled_stats=None):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        self.enabled_stats = enabled_stats or ['min', 'max']
        print(f"Statistics Alignment Layer initialized (learned-only). Enabled stats: {self.enabled_stats}")
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(pred_len, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        self.correctors = nn.ModuleDict()
        if 'min' in self.enabled_stats:
            self.correctors['min'] = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len))
        if 'max' in self.enabled_stats:
            self.correctors['max'] = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len))
        if 'trends' in self.enabled_stats:
            self.correctors['trends'] = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len))
        if 'mean' in self.enabled_stats:
            self.correctors['mean'] = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len))

        self.max_correctors = len(self.enabled_stats)
        if self.max_correctors > 0:
            self.correction_blender = nn.Sequential(
                nn.Linear(hidden_dim * (self.max_correctors + 1), self.max_correctors),
                nn.Sigmoid()
            )
            
    def forward(self, predictions, prompt_stats=None):
        # Learned-only statistical alignment (direct weight = 0)
        if not isinstance(prompt_stats, dict):
            return predictions
        statistical_items = prompt_stats.get('stats', None)
        if not statistical_items:
            return predictions

        batch_size, _, _ = predictions.shape
        corrected_predictions = predictions.clone()
        device = predictions.device

        # ensure modules on correct device
        if next(self.feature_extractor.parameters()).device != device:
            self.feature_extractor = self.feature_extractor.to(device)
            for stat in self.correctors:
                self.correctors[stat] = self.correctors[stat].to(device)
            if self.max_correctors > 0:
                self.correction_blender = self.correction_blender.to(device)

        batch_total_corrections = [torch.zeros_like(predictions[i, :, -1]) for i in range(batch_size)]

        for batch_idx, stats in statistical_items.items():
            if batch_idx >= batch_size:
                continue

            target_feature = -1
            target_sequence = predictions[batch_idx, :, target_feature]
            sequence_features = self.feature_extractor(target_sequence)

            applied_corrections = {}
            all_correction_features = {}

            for stat_name in self.enabled_stats:
                if stat_name not in stats:
                    continue
                target_value = torch.tensor([float(stats[stat_name])], device=device, dtype=target_sequence.dtype)

                # compute current stat and need for correction
                if stat_name == 'min':
                    current_stat = torch.min(target_sequence)
                    correction_needed = current_stat < target_value
                elif stat_name == 'max':
                    current_stat = torch.max(target_sequence)
                    correction_needed = current_stat > target_value
                elif stat_name == 'trends':
                    current_stat = target_sequence[-1] - target_sequence[0] if target_sequence.shape[0] >= 2 else torch.tensor(0.0, device=device, dtype=target_sequence.dtype)
                    correction_needed = True
                elif stat_name == 'mean':
                    current_stat = torch.mean(target_sequence)
                    correction_needed = torch.abs(current_stat - target_value) > 1e-4
                else:
                    continue

                if correction_needed and stat_name in self.correctors:
                    corrector_input = torch.cat([sequence_features, current_stat.unsqueeze(0), target_value], dim=-1)
                    learned_correction = self.correctors[stat_name](corrector_input)
                    applied_corrections[stat_name] = learned_correction
                    all_correction_features[stat_name] = self.feature_extractor(learned_correction)

            # fill zeros for missing stats to keep blender input stable
            for stat in self.enabled_stats:
                if stat not in all_correction_features:
                    all_correction_features[stat] = torch.zeros_like(sequence_features)

            if applied_corrections and self.max_correctors > 0:
                blend_input = torch.cat([sequence_features] + [all_correction_features[s] for s in self.enabled_stats], dim=-1)
                expected_dim = self.hidden_dim * (self.max_correctors + 1)
                if blend_input.shape[-1] != expected_dim:
                    print(f"Warning: Blender input dim mismatch! Expected {expected_dim}, got {blend_input.shape[-1]}. Skipping blend for item {batch_idx}.")
                    batch_total_corrections[batch_idx] = torch.zeros_like(target_sequence)
                else:
                    blend_weights = self.correction_blender(blend_input)
                    item_total = torch.zeros_like(target_sequence)
                    for i, stat in enumerate(self.enabled_stats):
                        if stat in applied_corrections:
                            item_total += applied_corrections[stat] * blend_weights[i]
                    batch_total_corrections[batch_idx] = item_total
            else:
                batch_total_corrections[batch_idx] = torch.zeros_like(target_sequence)

        final_corrections_tensor = torch.stack(batch_total_corrections, dim=0)
        if final_corrections_tensor.shape == corrected_predictions[:, :, -1].shape:
            corrected_predictions[:, :, -1] = corrected_predictions[:, :, -1] + final_corrections_tensor.to(corrected_predictions.dtype)
        else:
            print(f"Warning: Correction tensor shape mismatch. Pred: {corrected_predictions[:, :, -1].shape}, Corr: {final_corrections_tensor.shape}. Skipping batch correction.")

        return corrected_predictions

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x

class Model(nn.Module):
    def __init__(self, configs, device, full_ref_sequences=None, dataset_sizes=None):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len

        self.configs = configs

        # classifier-free guidance
        self.p_uncond = configs.p_uncond if hasattr(configs, 'p_uncond') else 0.3
        self.use_cfg = configs.use_cfg if hasattr(configs, 'use_cfg') else False
        self.guidance_scale = configs.guidance_scale if hasattr(configs, 'guidance_scale') else 2.0
        
        # stats alignment
        if hasattr(configs, 'use_alignment'):
            self.use_alignment = configs.use_alignment == 1
        else:
            self.use_alignment = False
        
        print(f"CFG: p_uncond={self.p_uncond}, use_cfg={self.use_cfg}, scale={self.guidance_scale}, use_alignment={self.use_alignment}")

        # reference sequences and sizes
        self.full_ref_sequences = full_ref_sequences
        self.dataset_sizes = dataset_sizes
        self.current_batch_idx = None
        self.current_split = None
        
        if hasattr(configs, 'alignment_type') and configs.alignment_type == 'sequence' and self.full_ref_sequences is not None:
             print(f"Model initialized with {len(self.full_ref_sequences)} pre-loaded reference sequences.")
             print(f"Dataset sizes: {self.dataset_sizes}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
    
        self.task_name = 'forecasting'
        print(f"Task name set to: {self.task_name}")
    
        # gpt2 backbone (time + text)
        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers] 
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config) 

        # word embedding (precomputed)
        word_embedding = torch.tensor(torch.load(configs.word_embedding_path, weights_only=False)).to(device=device)

        # freeze everything except norm, pos, lora
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = any(x in name for x in ['ln', 'wpe', 'lora'])
        for name, param in self.gpt2_text.named_parameters():
            param.requires_grad = 'wpe' in name

        # projections for time/text features
        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])

        # input/output layers
        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len)

        # blend param
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # move to device, set train mode
        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        self.device = device

        # statistics alignment layer
        enabled_stats = None
        if hasattr(configs, 'loss_stats'):
            enabled_stats = configs.loss_stats.split(',')
            print(f"Using statistics for alignment from args.loss_stats: {enabled_stats}")
        
        # learned-only alignment
        alignment_type = 'stats'
        
        self.stats_alignment = StatisticsAlignmentLayer(
            pred_len=configs.pred_len,
            hidden_dim=64,
            dropout=configs.dropout,
            enabled_stats=enabled_stats
        )

    def set_dataset_sizes(self, split, size):
        """Update dataset sizes after init."""
        if split in self.dataset_sizes:
            self.dataset_sizes[split] = size
            print(f"Updated dataset size for '{split}': {size}")
        else:
            print(f"Warning: Attempted to set size for unknown split '{split}'")

    def _calculate_batch_stats(self, batch_sequences_dict):
        """Compute stats for a batch: {idx: {stat: value}}."""
        batch_stats = {}
        stats_to_calc = self.configs.loss_stats.split(',')
        
        for idx, seq_data in batch_sequences_dict.items():
            item_stats = {}

            # convert to numpy array
            if isinstance(seq_data, torch.Tensor):
                seq = seq_data.detach().cpu().numpy()
            else:
                seq = np.array(seq_data)
            
            if seq.ndim > 1:
                seq = seq[..., -1].squeeze()
                
            if seq.size == 0:
                 print(f"Warning: Empty sequence encountered for batch index {idx} during stat calculation.")
                 continue

            try:
                if 'min' in stats_to_calc:
                    item_stats['min'] = np.min(seq)
                if 'max' in stats_to_calc:
                    item_stats['max'] = np.max(seq)
                if 'mean' in stats_to_calc:
                    item_stats['mean'] = np.mean(seq)
                if 'median' in stats_to_calc:
                    item_stats['median'] = np.nanmedian(seq)
                
                # filter out NaN results
                item_stats = {k: v for k, v in item_stats.items() if not np.isnan(v)}
                
            except Exception as e:
                 print(f"Error calculating stats for item {idx}: {e}")
                 item_stats = {}
    
            if item_stats:
                batch_stats[idx] = item_stats

        return batch_stats

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, force_cond=None):

        batch_ref_sequences = {}
        calculated_batch_stats = {'stats': {}, 'is_conditional': True}

        # Get batch size
        B = x_enc.size(0)
        
        # Determine conditioning status
        use_conditioning = True
        if force_cond is not None: use_conditioning = force_cond
        elif self.training and self.use_cfg: use_conditioning = torch.rand(1).item() > self.p_uncond
        calculated_batch_stats['is_conditional'] = use_conditioning

        # reference sequences -> stats for alignment/loss
        if self.full_ref_sequences is not None and self.current_batch_idx is not None and self.current_split in self.dataset_sizes:
            try:
                split_offset = 0
                if self.current_split == 'val': split_offset = self.dataset_sizes.get('train', 0)
                elif self.current_split == 'test': split_offset = self.dataset_sizes.get('train', 0) + self.dataset_sizes.get('val', 0)
                
                current_config_batch_size = self.configs.eval_batch_size if self.current_split == 'test' else self.configs.batch_size
                start_index = split_offset + self.current_batch_idx * current_config_batch_size
                end_index = start_index + B # Slice using actual batch size B

                if start_index < len(self.full_ref_sequences) and end_index <= len(self.full_ref_sequences):
                    sequences_np = self.full_ref_sequences[start_index:end_index]
                    for local_item_idx in range(B):
                         batch_ref_sequences[local_item_idx] = sequences_np[local_item_idx]
                    # compute stats from loaded references
                    calculated_stats = self._calculate_batch_stats(batch_ref_sequences)
                    calculated_batch_stats['stats'] = calculated_stats
                else:
                    if self.current_batch_idx == 0: 
                        print(f"WARN: Index [{start_index}:{end_index}] OOB for refs (len {len(self.full_ref_sequences)})")
            except Exception as e:
                print(f"Error extracting refs/stats: {e}")
        
        B, L, M = x_enc.shape

        # normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc - means, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x_enc - means) / stdev
        x = rearrange(x, 'b l m -> b m l')

        # encoder and transformer
        outputs_time1, outputs_text1 = self.in_layer(x)
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        # projections
        intermidiate_feat_time = tuple(self.time_proj[idx](feat) for idx, feat in enumerate(intermidiate_feat_time))
        intermidiate_feat_text = tuple(self.text_proj[idx](feat) for idx, feat in enumerate(intermidiate_feat_text))

        # output heads and reshape
        outputs_time = rearrange(self.out_layer(outputs_time[:, -M:, :]), 'b m l -> b l m')
        outputs_text = rearrange(self.out_layer(outputs_text[:, -M:, :]), 'b m l -> b l m')

        # denormalize and blend
        outputs_time = outputs_time * stdev + means
        outputs_text = outputs_text * stdev + means
        alpha = torch.sigmoid(self.alpha)
        combined_output = alpha * outputs_time + (1 - alpha) * outputs_text

        alignment_applied = False
        alignment_skip_reason = "None"
        
        # gating
        if not self.use_alignment:
            alignment_skip_reason = "use_alignment flag is False"
        elif not use_conditioning:
            alignment_skip_reason = "conditioning is disabled"
        else:
            alignment_input_data = {}
            alignment_type = getattr(self.configs, 'alignment_type', 'stats')
            if alignment_type == 'sequence' and batch_ref_sequences:
                sequences_for_alignment = {idx: torch.from_numpy(seq).to(combined_output.device, combined_output.dtype) 
                                        if isinstance(seq, np.ndarray) else seq.to(combined_output.device, combined_output.dtype)
                                        for idx, seq in batch_ref_sequences.items()}
                alignment_input_data = {'mode': 'sequence', 'sequences': sequences_for_alignment}
            elif alignment_type == 'stats' and calculated_batch_stats.get('stats'):
                alignment_input_data = {'mode': 'stats', 'stats': calculated_batch_stats['stats']}
            else:
                alignment_skip_reason = f"no valid alignment data for type '{alignment_type}'"
            
            if alignment_input_data:
                combined_output = self.stats_alignment(combined_output, alignment_input_data)
                alignment_applied = True
            elif alignment_skip_reason == "None":
                alignment_skip_reason = "no alignment data available"
        
        calculated_batch_stats['alignment_applied'] = alignment_applied
        calculated_batch_stats['alignment_skip_reason'] = alignment_skip_reason

        return combined_output, calculated_batch_stats

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, force_cond=None):
        """Forward with optional forced conditioning."""
        # CFG (conditional reasoning guidance) inference if enabled
        if not self.training and self.use_cfg and force_cond is None:
            dec_out, stats = self._classifier_free_guided_inference_with_stats(
                x_enc, x_mark_enc, x_dec, x_mark_dec, self.guidance_scale
            )
            return dec_out[:, -self.pred_len:, :], stats
        
        dec_out, stats = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, force_cond)
        return dec_out[:, -self.pred_len:, :], stats

    def _classifier_free_guided_inference_with_stats(self, batch_x, batch_x_mark, batch_y, batch_y_mark, guidance_scale=3.0):
        """CFG (conditional reasoning guidance) inference; returns predictions and stats from conditional pass."""
        # unconditional prediction
        uncond_pred, _ = self.forecast(
            batch_x, batch_x_mark, batch_y, batch_y_mark, 
            force_cond=False
        )
        
        # conditional prediction
        cond_pred, cond_stats = self.forecast(
            batch_x, batch_x_mark, batch_y, batch_y_mark, 
            force_cond=True
        )
        
        guided_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        
        # guided prediction
        return guided_pred, cond_stats