conf = {
        "WORK_PATH": "/home/wangming/GaitStrip/work/",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",

        "data": {
            'dataset_path': "/home/wangming/CASIAB_chuli",
            'resolution': '64',
            'dataset': 'CASIA-B',
            'pid_num': 74,
            'pid_shuffle': False,
        },
        "model": {
            'lr': 1e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (8, 8),
            'restore_iter': 0,
            'total_iter': 80000,
            'margin': 0.2,
            'num_workers': 0,
            'frame_num': 30,
            'hidden_dim': 256,
            'model_name': 'GaitStrip'



        }
    }
