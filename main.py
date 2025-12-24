from .regression import Manometer, YOLO, CropImage #type: ignore
from .utils import angle_to_percent, get_volume

yolo = YOLO("reader_manometer/runs/detect/train2/weights/best.pt")

def get_crop():
    crop = CropImage(
        yolo=yolo,
        imput_dir='./',
        output_dir='./crops'
    )
    crop.generate_crop('image2.jpg')


def get_vol():
    man = Manometer(
        yolo=yolo,
        regressor="reader_manometer/regressor.pt"
    )

    angles = man.get_angle(
        filename='./crops/image2_0.jpg'
    )


    if angles:
        
        print("ângulos:", angles)
        
        man_volume = angles[0]

        vol_percent = angle_to_percent(man_volume)
        print("porcentagem volume:", round(vol_percent, 2))

        print("volume:", round(get_volume(vol_percent, 800), 2))
