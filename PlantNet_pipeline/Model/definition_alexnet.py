"""
Alexnet.py 모델설정
"""
import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class defintion_model(nn.Module):
    def __init__(self, num_classes, class_counts=None, device='cpu'):
        super(defintion_model, self).__init__()
        self.device = device
        
        # AlexNet 로드
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.alexnet.classifier[-1] = nn.Linear(4096, num_classes)
        self.alexnet.to(self.device)
        
        # 손실함수 설정
        if class_counts:
            weights = [1.0 / c for c in class_counts.values()]
            weights_tensor = torch.FloatTensor(weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = None # 학습에서 초기화
        self._freeze_layers() # 초기 동결

    def _freeze_layers(self):
        """
        초기 학습 레이어설정
        """
        for param in self.alexnet.parameters():
            param.requires_grad = False
        for param in self.alexnet.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        nn상속
        """
        return self.alexnet(x)
        
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.alexnet.parameters())
    
    def unfreeze_step(self, stage, decay_factor, current_lr):
        """
        단계별 동결 해제 
        """
        if stage == 1:
            for param in self.alexnet.classifier[4].parameters(): param.requires_grad = True
            return current_lr * decay_factor
        elif stage == 2:
            for param in self.alexnet.classifier[1].parameters(): param.requires_grad = True
            return current_lr * decay_factor
        return current_lr