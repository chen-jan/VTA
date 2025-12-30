def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    # Handle top_k and num_kernels if they don't exist
    if hasattr(args, 'top_k') and hasattr(args, 'num_kernels'):
        print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    elif hasattr(args, 'top_k'):
        print(f'  {"Top k:":<20}{args.top_k:<20}')
    elif hasattr(args, 'num_kernels'):
        print(f'  {"Num Kernels:":<20}{args.num_kernels:<20}')
        
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    
    if hasattr(args, 'factor'):
        print(f'  {"Factor:":<20}{args.factor:<20}', end='')
    if hasattr(args, 'dropout'):
        print(f'{"Dropout:":<20}{args.dropout:<20}')
    else:
        print()
        
    if hasattr(args, 'embed'):
        print(f'  {"Embed:":<20}{args.embed:<20}', end='')
    if hasattr(args, 'activation'):
        print(f'{"Activation:":<20}{args.activation:<20}')
    else:
        print()
        
    if hasattr(args, 'output_attention'):
        print(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    
    if hasattr(args, 'des'):
        print(f'  {"Des:":<20}{args.des:<20}', end='')
    if hasattr(args, 'lradj'):
        print(f'{"Lradj:":<20}{args.lradj:<20}')
    else:
        print()
        
    if hasattr(args, 'use_amp'):
        print(f'  {"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    # Only print this section if the attributes exist
    if hasattr(args, 'p_hidden_dims') and hasattr(args, 'p_hidden_layers'):
        print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
        p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
        print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
        print()

    if hasattr(args, 'task_w') or hasattr(args, 'feature_w') or hasattr(args, 'output_w'):
        print("\033[1m" + "Loss Weights" + "\033[0m")
        task_w = args.task_w if hasattr(args, 'task_w') else "N/A"
        feature_w = args.feature_w if hasattr(args, 'feature_w') else "N/A"
        output_w = args.output_w if hasattr(args, 'output_w') else "N/A"
        print(f'  {"Feature Weight:":<20}{feature_w:<20}{"Output Weight:":<20}{output_w:<20}{"Task Weight:":<20}{task_w:<20}') 
        print()