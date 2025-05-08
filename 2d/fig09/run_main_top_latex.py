from reference import *
from bound import *
from dataset import *
from utils import *

basic_size = 14
uk_ticks = [0.1, 0.5]

rc_fonts = {
    "font.family": "Linux Libertine",
    "text.usetex": True,
    "font.size": basic_size,
    'text.latex.preamble':
        r"""
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """,
}

plt.rcParams.update(rc_fonts)

plt.figure(figsize=(12, 2.5))

plt.subplot(151)
plot_segment(p0, p1, n0, n1)
draw_rays(pL, p0, p1, n0, n1)
plot_segment_only([-1, -0.99], [2, 0])
plt.ylabel("$y$", rotation=0, labelpad=5, y=0.45)
plt.title("Configuration", fontsize=basic_size-1)

plt.subplot(152)
print(compute_ref(pL, p0, p1, n0, n1, 1))
print(compute_the_bound_recursive(pL, p0, p1, n0, n1, show=1))
plt.ylim(0.0, 0.7)
plt.ylabel("$u_k$", rotation=0, labelpad=10, y=0.45)
plt.yticks(uk_ticks)
plt.title("Position", fontsize=basic_size-1)

plt.subplot(153)
print(compute_ref(pL, p0, p1, n0, n1, 2))
print(compute_the_bound_recursive(pL, p0, p1, n0, n1, show=2))
plt.ylim(0, 3)
plt.ylabel("$E_k$", rotation=0, labelpad=15, y=0.45)
plt.title("Irradiance", fontsize=basic_size-1)

plt.subplot(154)
print(compute_the_bound_recursive(pL, p0, p1, n0, n1))
print(compute_ref(pL, p0, p1, n0, n1))
plt.xlim(0.0, 0.75)
plt.xticks(uk_ticks)
plt.ylim(0, 3)
plt.ylabel("$E_k$", rotation=0, labelpad=15, y=0.45)
plt.title("Joint (Path)", fontsize=basic_size-1)

plt.subplot(155)
ref_bins = np.array(ref_bins) + np.array(ref_bins2)
plt.plot(np.arange(0, 1, 0.0001), ref_bins, lw=1.6)
plt.xlim(0.0, 0.75)
plt.xticks(uk_ticks)
plt.ylim(0, 3)
plt.ylabel("$E_k$", rotation=0, labelpad=15, y=0.45)
plt.title("Joint (Tuple)", fontsize=basic_size-1)
m = 1
for i in range(10000):
    plt.fill_between([i * 1e-4, (i+1) * 1e-4], 0.0, [bound_bins[i]*m, bound_bins[i]*m], facecolor=(154/255, 200/255, 219/255), alpha=0.4)

plt.subplots_adjust(left=0.048, right=0.988, bottom=0.1, top=0.85, wspace=0.5)
    
plt.savefig("bound_top.pdf")
plt.show()