import tensorflow as tf
import os
from scipy.spatial.transform import Rotation
import numpy as np

'''Base DatasetsLoader Class'''
class BaseDatasetsLoader:
    def __init__(self,name='BaseDatasetsLoader'):
        self.name = name
    def MakeTrainDatasets(self):
        print("make traindatasets")
    def MakeTestDatasets(self):
        print("make testdatasets")
    def GetTrainDatasets(self):
        print("get traindatasets")
    def GetTestDatasets(self):
        print("get testdatasets")

class TfRecordDatasetsLoader(BaseDatasetsLoader):
    def __init__(self, Origin_FIle_Path=[],Save_File_Path=[],name='TfRecordDatasetsLoader'):
        super(TfRecordDatasetsLoader,self).__init__(name=name)
        if len(Save_File_Path)==0:
                raise Exception("The Save_File_Path is not correct,Please Check it!",Save_File_Path)
        if len(Origin_FIle_Path)>1:
            print("Load Train and Test FIle\r\n")
            self.Origin_Train_FIle_Path = Origin_FIle_Path[0]
            self.Origin_Test_File_Path = Origin_FIle_Path[1]
            self.Save_Train_FIle_Path = Save_File_Path[0]
            self.Save_Test_FIle_Path = Save_File_Path[1]
            if not (os.path.isdir(self.Save_Train_FIle_Path)) or  not (os.path.isdir(self.Save_Test_FIle_Path)):
               os.makedirs(self.Save_Train_FIle_Path)
               os.makedirs(self.Save_Test_FIle_Path)
        elif len(Origin_FIle_Path)==1:
            print("Load Train  FIle\r\n")
            self.Origin_Train_FIle_Path = Origin_FIle_Path[0]
            self.Save_Train_FIle_Path = Save_File_Path[0]
            if not (os.path.isdir(self.Save_Train_FIle_Path)):
                os.makedirs(self.Save_Train_FIle_Path)
        elif len(Origin_FIle_Path)==0:
            raise Exception("The Origin_FIle_Path is not correct,Please Check it!",Origin_FIle_Path)

    def MakeTrainDatasets(self):
        '''rewrite your traindatasets make function,in this example, I rewrite this function for my own mvs-datasets'''
        def ParseTxtFunc(txt_file):
            img_name_pose_dict = dict() 
            pair_info = open(txt_file, "r")
            pair_file_context = pair_info.readline()
            line_count = 0
            while pair_file_context:
                pair_file_context = pair_info.readline()
                if len(pair_file_context)>12:
                    line_count = line_count+1
                    context_splits = pair_file_context.split(" ")
                    img_name = context_splits[0]
                    img_pose_R = np.array([ float(i) for i in context_splits[10:-3] ]).reshape((3,3))
                    img_pose_t = np.array([float(i) for i in context_splits[-3:]])
                    img_pose_quart = Rotation.from_matrix(img_pose_R).as_quat()
                    img_pose = np.concatenate([img_pose_t,img_pose_quart])
                    img_name_pose_dict[img_name]=img_pose
                    print("readline-->",pair_file_context.split(" "))
                    print("type-->",type(context_splits))
                    print("img_name-->",img_name)
                    print("img_pose_len-->",len(img_pose_t))
                    print("img_pose_quart-->",img_pose_t,img_pose_quart)
                    print("img_pose--->",img_pose)
            print(img_name_pose_dict.keys())
            print(img_name_pose_dict.values())
            return img_name_pose_dict
        #1.read image_label
        train_labels = []
        for home_dirs,dir_lists,file_lists in os.walk(self.Origin_Train_FIle_Path):
            #1.get labels from folder name
            if home_dirs ==self.Origin_Train_FIle_Path:
                if len(dir_lists)>=1:
                    for sub_dir in dir_lists:
                        if sub_dir.find("tfrecords")<0:
                            train_labels.append(sub_dir)
        #2.read image and pose
        for per_label in train_labels:
                train_data_file_path = os.path.join(self.Origin_Train_FIle_Path,per_label)
                #print("current traindata_path-->",train_data_file_path)
                for _,_,file_lists in os.walk(train_data_file_path):
                    print("len_file_lists",len(file_lists))
                    for file in file_lists:
                        if file.find("_par")>0:
                            train_data_file_path = os.path.join(train_data_file_path,file)
                            train_data_img_pose_dict = ParseTxtFunc(train_data_file_path)
                            


                
            
            


        



    



#########################################Test Module#######################################################
if __name__ =="__main__":
    m_tfrecorddataloader = TfRecordDatasetsLoader(["/home/swx/m_mode_datasets/temp_and_dino_datasets"],["/home/swx/m_mode_datasets/temp_and_dino_datasets/traindatasets_tfrecords"])
    m_tfrecorddataloader.MakeTrainDatasets()