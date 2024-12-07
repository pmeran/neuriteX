
# © 2024 Paul Meraner
# neuriteX

from neuriteX_methods import *

n_j = 8 # process number, must match with <cpu_count> in bash script 'segmentation.sh', line 'SBATCH -c <cpu_count>'
# get logical cpu count with bash command psutil.cpu_count()

path_ori = './img_ori/'
path_corr  = './img_corr/'

img_list = [os.path.basename(x) for x in glob.glob('{}*.png'.format(path_corr))]
img_list = sorted(img_list)

def seg_parallel(im_fname):
    print(im_fname)
    im = cv2.imread(path_corr + im_fname, -1)
    D = nX_segmentation(im, im_fname.replace('_corr.png', '.png'), ptUL=(100,100), eH=100, extF=3, win=3, ord=2, t=100000)
    return D

results_seg = Parallel(n_jobs = n_j)(delayed(seg_parallel)(img_fname) for img_fname in img_list)

df_seg = pd.DataFrame(results_seg)
df_seg.to_pickle('./df_seg.pkl')







