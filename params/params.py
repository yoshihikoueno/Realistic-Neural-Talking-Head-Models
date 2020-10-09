#K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape, path_to_mp4

#number of frames to load
K = 8

#path to main weight
path_to_chkpt = 'model_weights.tar' 

#path to backup
path_to_backup = 'backup_model_weights.tar'

#CHANGE first part
#path_to_Wi = "/dataset/"+"Wi_weights"
path_to_Wi = "Wi_weights"
#path_to_Wi = "test/"+"Wi_weights"

#CHANGE if better gpu
batch_size = int(2.5 * 2)

#dataset save path
path_to_preprocess = '/dataset/preprocessed/'

#default for Voxceleb
frame_shape = 224

#path to dataset
path_to_mp4 = '/dataset/dev/mp4'

# VGG19CAFFE_weight_path
VGG19_caffe_weight_path = 'vgg19-d01eb7cb.pth'
