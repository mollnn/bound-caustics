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
    return np.minimum(imga, 1e2) 
figure = CropComparison(
    reference_image=readexr("results/fig_plane_Ref.exr"),
    method_images=[
        readexr("results/fig_plane_cs_gamma100.exr"),
        readexr("results/fig_plane_cs_one.exr"),
        readexr("results/fig_plane_cs_multiuni.exr"),
        readexr("results/fig_plane_cs_enum.exr"),
    ],
    crops=[
        Cropbox(top=260, left=1013, height=60, width=60, scale=5),
        Cropbox(top=410, left=500, height=40, width=40, scale=5),
    ],
    scene_name="Plane (25 sec)",
    method_names=["Reference","Ours ", "One-sample ", "Uniform P=0.04 ", "Enumerate"],
    spp_list= [33, 54, 16, 1]
)
figuregen.figure([figure.figure_row], width_cm=17, filename="results/cmp_sample_plane.pdf")
os.system("results\\cmp_sample_plane.pdf")
# Please modify the figuregen template!!!
# @property
# def error_metric_name(self) -> str:
#     return "RelMSE"
# def compute_error(self, reference_image, method_image) -> Tuple[str, List[float]]:
        # return image.relative_mse_outlier_rejection(method_image, reference_image, 0.01, 0.1)