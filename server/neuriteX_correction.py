
# © 2024 Paul Meraner
# neuriteX

from neuriteX_methods import *

n_j = 8 # process number, must match with <cpu_count> in bash script 'correction.sh', line 'SBATCH -c <cpu_count>'
# get logical cpu count with bash command psutil.cpu_count()

path_ori = './img_ori/'
path_corr  = './img_corr/'

if not os.path.exists('./img_corr/'):
    os.makedirs('./img_corr/')
else:
    pass

img_list = [os.path.basename(x) for x in glob.glob('{}*.png'.format(path_ori))]
img_list = sorted(img_list)

# original server code
def corr_parallel(im_fname):
    print(im_fname)
    im = cv2.imread(path_ori + im_fname, -1)
    im_corr, exit_status = nX_correction(im, promMin = 5, perc_hi = 95, lim_I_new = 180, gamma = 0.8, win = 3, ord = 2)
    im_fname_new = im_fname.replace('.png', '_corr.png')
    cv2.imwrite(path_corr + im_fname_new, im_corr)
    return [im_fname_new, exit_status]

results_corr = Parallel(n_jobs = n_j)(delayed(corr_parallel)(img_fname) for img_fname in img_list)

df_corr = pd.DataFrame(results_corr)
df_corr.to_pickle('./df_corr.pkl')



