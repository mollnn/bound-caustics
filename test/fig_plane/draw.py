import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import simpleimageio as sio
# import figuregen
# from figuregen.util.templates import *
# from figuregen.util.image import Cropbox
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myfiguregen.util.templates import *
from myfiguregen.util.image import Cropbox

def readexr(a, crop = {}):
    imga = cv2.imread(a, cv2.IMREAD_UNCHANGED).astype("float")[:, :, [2, 1, 0]]
    imga = np.nan_to_num(imga)
    return np.minimum(imga, 1e4) 

figure = CropComparison(
    reference_image=readexr("results/fig_plane_Ref.exr"),
    method_images=[
        readexr("results/fig_plane_Bounded.exr"),
        readexr("results/fig_plane_Enum.exr"),
        readexr("results/fig_plane_MPG.exr"),
        readexr("results/fig_plane_SMS.exr"),
        readexr("results/fig_plane_UPSMCMC.exr"),
        # readexr("results/fig_plane_SPPM.exr"),
    ],
    crops=[
        Cropbox(top=288, left=1026, height=60, width=60, scale=5),
        Cropbox(top=440, left=440, height=60, width=60, scale=5),
    ],
    scene_name="Plane (30 sec)",
    method_names=["Reference", "Ours",  "SP", "MPG", "SMS", "UPSMCMC"],
    spp_list= [40, 1, 2, 2, 21, 165]
)
figuregen.figure([figure.figure_row], width_cm=19, filename="results/fig_plane.pdf")
os.system("results\\fig_plane.pdf")
# Please modify the figuregen template!!!
# @property
# def error_metric_name(self) -> str:
#     return "RelMSE"

# def compute_error(self, reference_image, method_image) -> Tuple[str, List[float]]:
        # return image.relative_mse_outlier_rejection(method_image, reference_image, 0.01, 0.1)
