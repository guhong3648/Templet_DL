import os

from utils.utils import make_dir

class HP():
    def __init__(self, name='temp', device='cuda'):
        self.name = name
        self.device = device
        self.multi_gpu = False
        
        self.epochs = 200
        self.batch_size = 36
        self.optimizer_lr = 1e-4
        self.scheduler_step = 10
        self.scheduler_gamma = 0.8
        self.monitoring_cycle = 1
        self.save_cycle = 100
        
        self.batch_size_inference = 12
        self.epoch_load = None
        
        # Path
        self.path_main = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path_data          = f'{self.path_main}/data'
        self.path_Res           = f'{self.path_main}/res'
        self.path_res           = f'{self.path_main}/res/{self.name}'
        self.path_model         = f'{self.path_main}/res/{self.name}/model'
        self.path_inference     = f'{self.path_main}/res/{self.name}/inference'
        self.path_evaluation    = f'{self.path_main}/res/{self.name}/evaluation'
        self.path_fig           = f'{self.path_main}/res/{self.name}/fig'
        self.path_temp          = f'{self.path_main}/res/{self.name}/temp'
        
        make_dir(self.path_Res)
        make_dir(f'{self.path_Res}/Fig')
        make_dir(self.path_res)
        make_dir(self.path_model)
        make_dir(self.path_inference)
        make_dir(self.path_evaluation)
        make_dir(self.path_fig)
        make_dir(self.path_temp)
        
hp = HP(name='temp')

        
