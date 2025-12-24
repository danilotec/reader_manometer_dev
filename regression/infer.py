import torch
import cv2
from ultralytics.models import YOLO
from .model import AngleRegressor

class Manometer:
    def __init__(self, yolo: YOLO, regressor: str) -> None: 
        self.yolo = yolo
        self.reg = AngleRegressor()
        self.reg.load_state_dict(torch.load(regressor))
        self.reg.eval()

    def get_angle(self, filename: str) -> list[float] | None:
        img = cv2.imread(filename)
        if img is not None:
            result = self.yolo(img)[0]
            result.plot()
            self.angles = []

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (224, 224))

                t = torch.tensor(crop).float().permute(2, 0, 1).unsqueeze(0) / 255

                ang_norm = self.reg(t).item()
                self.angles.append(ang_norm * 360)
            return self.angles
        return None
        
    def angle_to_percent(self, angle: float) -> float:
        '''
        :param angle: esse paramentro deve ser preenchido com o retorno do metodo get_angle da classe manometer
        :type angle: float
        
        :return: retorna o angulo do ponteiro do manometro
        :rtype: float
        '''

        if angle < 135:
            angle += 360

        percentual = (angle - 135) / 270
        return max(0.0, min(1.0, percentual))
    
    def get_volume(self, scale: float) -> float:
        '''    
        :param percent: se refere a porcentagem do circulo do manometro
        esse valor deve vir da função angle_to_percent 
        :type percent: float
        
        :param scale: se refere ao valor maximo de leitura do manometro, que reprezenta o 100%
        do mesmo
        :type scale: float
        
        :return: retorna o valor real do manometro baseado na posição do ponteiro
        :rtype: float
        '''
        return self.angle_to_percent(self.angles[0]) * scale
