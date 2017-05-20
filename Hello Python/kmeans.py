from scipy.misc import imread,imshow,imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from functools import partial
from optparse import OptionParser

def kmeans(img,K,epsilon):

    img = img.astype(np.float64)

    randpos = partial(np.random.randint,0,min(img.shape[0],img.shape[1]))
    cpos = randpos(2*K).reshape(2,K)
    cx = cpos[0]
    cy = cpos[1]

    center = img[cx,cy]

    # shape 1,220,220,3 - k,1,1,3
    img = img.reshape(1,img.shape[0],img.shape[1],3)
    center = center.reshape(K,1,1,3)
    dis = (img-center)**2

    img = np.squeeze(img)
    center = np.squeeze(center)

    tsum = np.sum(dis,axis=3)

    pos_label = tsum.argmin(axis=0)

    pre_center = np.sum(center)
    diff = np.inf
    # ite = 0

    while(diff>epsilon):

        for i in range(K):
            center[i] = np.mean(img[pos_label == i],axis=0)
            img[pos_label == i] = center[i]

        current_center = np.sum(center)
        diff = np.abs(current_center-pre_center)
        pre_center = current_center
        # ite+=1

    # print(diff)
    # print(ite)

    img = img.astype(np.float16)
    return img

I = 10
def inter_vis(img):

    def draw_button():

        global button_sub, button_plus
        point_sub = plt.axes([0.3, 0.03, 0.1, 0.03])
        point_plus = plt.axes([0.6, 0.03, 0.1, 0.03])
        button_sub = Button(point_sub, "-")
        button_sub.on_clicked(on_press_sub)
        button_plus = Button(point_plus, "+")
        button_plus.on_clicked(on_press_plus)

    def on_press_plus(event):

        if event.inaxes == None:
            print("none")
            return
        global I
        I = I + 1
        fig = event.inaxes.figure
        ax1 = fig.add_subplot(111)
        nimg = kmeans(img, I, 1e-4)
        nimg = nimg.astype(np.uint8)
        plt.title("K = " + str(I))
        ax1.imshow(nimg)
        plt.axis("off")
        fig.canvas.draw()

    def on_press_sub(event):

        if event.inaxes == None:
            print("none")
            return
        global I
        if(I==2):
            print("at least 2 center!")
            return

        I = I - 1

        fig = event.inaxes.figure
        ax1 = fig.add_subplot(111)
        nimg = kmeans(img, I, 1e-4)
        nimg = nimg.astype(np.uint8)
        plt.title("K = " + str(I))
        ax1.imshow(nimg)
        plt.axis("off")
        fig.canvas.draw()


    plt.ioff()
    fig = plt.figure()
    draw_button()

    ax1 = fig.add_subplot(111)
    nimg = kmeans(img, I, 1e-4)
    nimg = nimg.astype(np.uint8)
    plt.title("K = " + str(I))
    ax1.imshow(nimg)
    plt.axis("off")
    plt.show()



def visualize(img,start=2,end=100):

    plt.ion()
    plt.axis("off")

    for i in range(start,end):

        nimg = kmeans(img, i, 1e-4)
        nimg = nimg.astype(np.uint8)
        plt.title("K = " + str(i))
        plt.imshow(nimg)
        plt.pause(0.01)

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-v", "--visualize", action="store_true",
                      dest="visualize",
                      default=False,
                      help="visualize kmeans automatically from 2 to 100")
    parser.add_option("-i", "--interactive", action="store_true",
                      dest="interactive",
                      default=False,
                      help="use interactive mode to change K")

    img = np.floor(imread("/home/ryan/Desktop/cat.jpg"))

    (options, args) = parser.parse_args()

    if options.visualize == True:
        visualize(img)

    if options.interactive == True:
        inter_vis(img)

    if (options.visualize == False) and (options.interactive == False):
        visualize(img)