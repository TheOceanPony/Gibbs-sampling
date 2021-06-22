import sys 
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt


def initial_image(h, w, p, pc):
    
    img = np.zeros((h,w), dtype=np.uint8)

    # Lines
    r=np.random.binomial(1,pc,size=(h))
    c=np.random.binomial(1,pc,size=(w))
    for i in range(0,h):
        for j in range(0,w):
            img[i, j] = max(r[i], c[j])
            
    # Noise
    noise=np.random.binomial(1,p,size=(h,w))

    return img, img^noise, r,c


def reconstruct(r,c):
    h,w = r.shape[0], c.shape[0]
    img = np.zeros((h,w), dtype=np.uint8)

    # Lines
    for i in range(0,h):
        for j in range(0,w):
            img[i, j] = max(r[i], c[j])

    return img


def bernoulli(p):
    v=random.uniform(0,1)
    return int(v<p)


def calculate_probab(im,h,w,p,a_pr,fixed_lines):

    prob=np.zeros(h)
    pos=np.arange(2)

    for i in range(0,h):
        g = (a_pr ** np.arange(2))*((1-a_pr)**1-np.arange(2))
        for j in range(0,w):
            g = g*(((im[i,j]^(np.logical_or(fixed_lines[j],pos)))*p)+((im[i,j]^(1-np.logical_or(fixed_lines[j],pos)))*(1-p)))
        g=(g/np.sum(g))
        prob[i]=g[1]

    return prob


def gibbs_sampler(new_im,p,pc,iters):

    h=new_im.shape[0]
    w=new_im.shape[1]
    r=np.random.randint(2,size=h)

    for i in tqdm(range(iters)):
        c=generate_lines(calculate_probab(np.transpose(new_im),w,h,p=p,a_pr=pc,fixed_lines=r))
        r=generate_lines(calculate_probab(new_im,h,w,p,pc,c))

    return r,c


def exact_answ(im,h,w,pc,p):
    var=np.arange(2)
    rows_prob=np.zeros((2))
    prob=np.zeros((2,w))
    p_apr=pc**np.arange(2)*(1-pc)**(1-np.arange(2))
    cols=np.zeros((w))
    for c in var:
        for i in range(0,w):
            for r in var:
                rows_prob[r]=p_apr[r]
                for j in range(0,h):
                    rows_prob[r]*=(im[j,i]^(c or r))*p+(im[j,i]^(1-(c or r)))*(1-p)
            prob[c,i]=np.sum(rows_prob)
    
    for i in tqdm(range(0,w)):
        prob[:,i]=prob[:,i]/np.sum(prob[:,i])
        cols[i]=bernoulli(prob[1,i])
    rows=generate_lines(calculate_probab(im,h,w,p,p_apr,cols))
    return rows,cols


def generate_lines(prob):

    num=prob.shape[0]
    ind=[]
    res=np.zeros(num)

    for i in range(0,num):
        res[i]=bernoulli(prob[i])
        if(res[i]):
            ind.append(i)

    return res


def main(h, w, p, pc, n_iter):



    id_img, img,rows, cols = initial_image(h, w, p, pc)

    print("Gibbs sampler:")
    print("=================================================")
    [res_r, res_c] = gibbs_sampler(img,p,pc,n_iter)
    
    r_acc, c_acc = (res_r==rows).sum(),(res_c==cols).sum()
    print(f"Result accuracy: {((r_acc/h)+(c_acc/w))/2 }")

    plt.imsave("img.png",img)
    plt.imsave("img_ideal.png", id_img)
    plt.imsave("img_reconstructed.png",reconstruct(res_r,res_c))
    print(f"Results written to disk.")

    print("")
    print("Exact solution:")
    print("=================================================")
    [res_r, res_c] = exact_answ(img,h,w,pc,p)
    r_acc, c_acc = (res_r==rows).sum(),(res_c==cols).sum()
    print(f"Result accuracy: {((r_acc/h)+(c_acc/w))/2 }")
    plt.imsave("img_exact.png",reconstruct(res_r,res_c))
    print(f"Results written to disk.")


def print_inp_error():
        print("USAGE : gibbs.py [height] [width] [p_noise] [p_line] [n_iter]")
        print("    [height] and [width] - image shape [int]")
        print("    [p_noise]            - decimal in  [0,1]")
        print("    [p_line]             - decimal in  [0,1]")
        print("    [n_iter]             - number of iterations")


if __name__ == "__main__":

    # Input check
    if len(sys.argv) < 6:
        print_inp_error()
        sys.exit(1)

    else:
        # input str check
        h,w = int(sys.argv[1]), int(sys.argv[2])

        # noise check
        p = float( sys.argv[3] )
        if p < 0 or p > 1:
            print_inp_error()
            sys.exit(1)

        pc = float( sys.argv[4] )
        if pc < 0 or pc > 1:
            print_inp_error()
            sys.exit(1)

        n_iter = int(sys.argv[5])
    
    print(f"Runnig...")
    main(h, w, p, pc, n_iter)
