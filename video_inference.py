import torch
import cv2
from matplotlib import pyplot as plt
from threading import Thread
from queue import Queue

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from webcam_demo.webcam_extraction_conversion import *

from params.params import path_to_chkpt
from tqdm import tqdm

"""Init"""

#Paths
#path_to_model_weights = 'finetuned_model.tar'
path_to_model_weights = 'model_weights.tar'
path_to_embedding = 'e_hat_video.tar'
#path_to_mp4 = 'test_vid2.webm'
path_to_mp4 = 'reference.mp4'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_model_weights, map_location=cpu) 
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=False, e_finetuning=e_hat)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)


"""Main"""
print('PRESS Q TO EXIT')
cap = cv2.VideoCapture(path_to_mp4)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
i = 0
size = (256*3,256)
#out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc('M','P','4','2'), 30, size)
video = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

in_queue = Queue(12)
out_queue = Queue(12)

def feeder(dst_queue):
    cap = cv2.VideoCapture(path_to_mp4)
    device = torch.device("cuda:0")
    while True:
        x, g_y, remaining = generate_landmarks(cap=cap, device=device, pad=50)
        if not remaining: break
        dst_queue.put((x, g_y))
    cap.release()
    dst_queue.put((None, None))
    return

def saver(src_queue, out_stream):
    while True:
        img = src_queue.get()
        out_stream.write(img)
        src_queue.task_done()
    return

Thread(target=feeder, args=(in_queue,), daemon=True).start()
Thread(target=saver, args=(out_queue, video), daemon=True).start()

with torch.no_grad():
    for i in tqdm(range(n_frames)):
        x, g_y = in_queue.get()
        if x is None: break
        g_y = g_y.unsqueeze(0)/255
        x = x.unsqueeze(0)/255

        x_hat = G(g_y, e_hat)

        plt.clf()
        out1 = x_hat.transpose(1,3)[0]
        out1 = out1.to(cpu).numpy()
        out2 = x.transpose(1,3)[0]
        out2 = out2.to(cpu).numpy()
        out3 = g_y.transpose(1,3)[0]
        out3 = out3.to(cpu).numpy()

        fake = cv2.cvtColor(out1*255, cv2.COLOR_BGR2RGB)
        me = cv2.cvtColor(out2*255, cv2.COLOR_BGR2RGB)
        landmark = cv2.cvtColor(out3*255, cv2.COLOR_BGR2RGB)
        img = np.concatenate((me, landmark, fake), axis=1)
        img = img.astype('uint8')
        out_queue.put(img)

out_queue.join()
cap.release()
video.release()
