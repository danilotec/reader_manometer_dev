from .regression import Manometer, YOLO, CropImage
import asyncio

yolo = YOLO("reader_manometer/runs/detect/train2/weights/best.pt")

def create_crop(image_path: str, crop_class: int) -> list[str] | None:
    ''' O crop_class deve ser um int, as opções são: 0 ou 1
    onde: 
        0 -> gera o crop do manometro inteiro da imagem
        1 -> gera o crop do ponteiro do manometro
    retorna uma lista de nomes de imagens 
    ou retorna None caso o cv2.imread nao retorne o MatLike
    '''

    crop = CropImage(
        yolo=yolo,
        imput_dir='./',
        output_dir='./crops'
    )
    return crop.generate_crop(image_path, crop_class)


def process_volum(crop_image_path: str) -> tuple:
    ''' 
    Retorna uma Tupla(list[angles], volum)
    '''
    man = Manometer(
        yolo=yolo,
        regressor="reader_manometer/regressor.pt"
    )
    angles = man.get_angle(filename=crop_image_path)

    if angles:
        volum = round(man.get_real_value(800), 2)
        return angles, volum
    
    return None, None

async def get_crops_names(image_path: str, crop_class: int) -> list[str] | None:
    crop_names = await asyncio.to_thread(lambda: create_crop(image_path, crop_class))

    if crop_names:
        return crop_names
    return None


async def get_angles_volum(crop_image_path: str) -> tuple:
    angles, volum = await asyncio.to_thread(lambda: process_volum(crop_image_path))

    if angles:
        return angles, volum
    return None, None

