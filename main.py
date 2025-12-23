from regression import Manometer
from utils import angle_to_percent, get_volume

man = Manometer(
    model="runs/detect/train2/weights/best.pt",
    regressor="regressor.pt"
    )

angles = man.get_angle(
    filename="dataset/raw_images/image3.jpeg"
    )


if angles: 
    print('angulos: ', angles)
    
    man_pression = angles[0]
    man_volume = angles[1]

    percent = angle_to_percent(man_pression)
    print('porcentagem: ', round(percent, 2))

    print('pressão: ', round(get_volume(percent, 25), 2))

    vol_percent = angle_to_percent(man_volume)
    print('porcentagem: ', round(vol_percent, 2))

    print('volume: ', round(get_volume(percent, 800), 2))
    