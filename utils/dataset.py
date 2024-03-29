import torch
from torch.utils.data import Dataset
import scipy.sparse


# ----- Multimodal and Multilingual (M3L) -----
class M3LDataset(Dataset):

    """Class to load BoW and the contextualized embeddings for *aligned multilingual* datasets"""

    def __init__(self, X_contextual, X_bow, X_image_emb, idx2token, num_lang=2, is_inference=False):
        # during training, data is multilingual AND multimodal
        # during inference, data is monolingual AND monomodal (either image or text)
        if is_inference is False:
            if X_bow[0].shape[0] != X_contextual[0].shape[0]:
                raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                                "You might want to check if the BoW preparation method has removed some documents. ")
        # else:
        #     if X_bow.shape[0] != X_contextual.shape[0]:
        #         raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
        #                         "You might want to check if the BoW preparation method has removed some documents. ")
        #
        # if labels is not None:
        #     if labels.shape[0] != X_bow.shape[0]:
        #         raise Exception(f"There is something wrong in the length of the labels (size: {labels.shape[0]}) "
        #                         f"and the bow (len: {X_bow.shape[0]}). These two numbers should match.")

        self.X_bow = X_bow
        self.X_contextual = X_contextual
        self.X_image = X_image_emb
        self.idx2token = idx2token
        self.num_lang = num_lang
        self.inference_mode = is_inference

    def __len__(self):
        """Return length of dataset."""
        # during training, X_bow, X_contextual and X_image are all available
        if self.inference_mode is False:
            return self.X_contextual[0].shape[0]
        # during inference, either X_contextual or X_image is available, not both
        else:
            if self.X_contextual is not None:
                return self.X_contextual.shape[0]
            else:
                return self.X_image.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        # TRAINING: dataset is multimodal AND multilingual (X_contextual will have 1 extra row for the image embedding)
        if self.inference_mode is False:
            if type(self.X_bow[0][i]) == scipy.sparse.csr.csr_matrix:
                X_bow_collect = []
                X_contextual_collect = []
                for l in range(self.num_lang):
                    X_bow = torch.FloatTensor(self.X_bow[l][i].todense())
                    X_contextual = torch.FloatTensor(self.X_contextual[l][i])
                    X_bow_collect.append(X_bow)
                    X_contextual_collect.append(X_contextual)
                # X_bow_collect: L x vocab_size
                X_bow_collect = torch.stack(X_bow_collect)
                # X_contextual_collect: L x bert_dim
                X_contextual_collect = torch.stack(X_contextual_collect)
                # X_image: bert_dim
                X_image = torch.FloatTensor(self.X_image[i])
            else:
                X_bow_collect = []
                X_contextual_collect = []
                for l in range(self.num_lang):
                    X_bow = torch.FloatTensor(self.X_bow[i])
                    X_contextual = torch.FloatTensor(self.X_contextual[i])
                    X_bow_collect.append(X_bow)
                    X_contextual_collect.append(X_contextual)
                # X_bow_collect: L x vocab_size
                X_bow_collect = torch.stack(X_bow_collect)
                # X_contextual_collect: L x bert_dim
                X_contextual_collect = torch.stack(X_contextual_collect)
                # X_image: bert_dim
                X_image = torch.FloatTensor(self.X_image[i])
            return_dict = {'X_bow': X_bow_collect, 'X_contextual': X_contextual_collect, 'X_image': X_image}
        # INFERENCE: dataset is monolingual AND monomodal (either text or image)
        else:
            # X_bow is just a dummy variable
            X_bow = torch.FloatTensor(torch.rand(10))
            if self.X_contextual is not None:
                X_test = torch.FloatTensor(self.X_contextual[i])
            else:
                X_test = torch.FloatTensor(self.X_image[i])
            return_dict = {'X_contextual': X_test, 'X_bow': X_bow}
        return return_dict



# ----- Multilingual -----
class PLTMDataset(Dataset):

    """Class to load BoW and the contextualized embeddings for *aligned multilingual* datasets"""

    def __init__(self, X_contextual, X_bow, idx2token, labels=None, num_lang=2, is_inference=False):

        # if we are in training mode, dataset is multilingual; inference is monolingual
        if is_inference is False:
            if X_bow[0].shape[0] != X_contextual[0].shape[0]:
                raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                                "You might want to check if the BoW preparation method has removed some documents. ")
        else:
            if X_bow.shape[0] != X_contextual.shape[0]:
                raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                                "You might want to check if the BoW preparation method has removed some documents. ")

        if labels is not None:
            if labels.shape[0] != X_bow.shape[0]:
                raise Exception(f"There is something wrong in the length of the labels (size: {labels.shape[0]}) "
                                f"and the bow (len: {X_bow.shape[0]}). These two numbers should match.")

        self.X_bow = X_bow
        self.X_contextual = X_contextual
        self.idx2token = idx2token
        self.labels = labels
        self.num_lang = num_lang
        self.inference_mode = is_inference

    def __len__(self):
        """Return length of dataset."""
        if self.inference_mode is False:
            return self.X_contextual[0].shape[0]
        else:
            return self.X_contextual.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        # TRAINING: dataset is multilingual
        if self.inference_mode is False:
            if type(self.X_bow[0][i]) == scipy.sparse.csr_matrix:
                X_bow_collect = []
                X_contextual_collect = []
                for l in range(self.num_lang):
                    X_bow = torch.FloatTensor(self.X_bow[l][i].todense())
                    X_contextual = torch.FloatTensor(self.X_contextual[l][i])
                    X_bow_collect.append(X_bow)
                    X_contextual_collect.append(X_contextual)
                X_bow_collect = torch.stack(X_bow_collect)
                X_contextual_collect = torch.stack(X_contextual_collect)
            else:
                X_bow_collect = []
                X_contextual_collect = []
                for l in range(self.num_lang):
                    X_bow = torch.FloatTensor(self.X_bow[i])
                    X_contextual = torch.FloatTensor(self.X_contextual[i])
                    X_bow_collect.append(X_bow)
                    X_contextual_collect.append(X_contextual)
                X_bow_collect = torch.stack(X_bow_collect)
                X_contextual_collect = torch.stack(X_contextual_collect)

            return_dict = {'X_bow': X_bow_collect, 'X_contextual': X_contextual_collect}
        # INFERENCE: dataset is monolingual
        else:
            # we don't care about X_bow during inference
            if type(self.X_bow[i]) == scipy.sparse.csr.csr_matrix:
                 X_bow = torch.FloatTensor(self.X_bow[i].todense())
                 X_contextual = torch.FloatTensor(self.X_contextual[i])
            else:
                X_bow = torch.FloatTensor(self.X_bow[i])
                X_contextual = torch.FloatTensor(self.X_contextual[i])
            return_dict = {'X_bow': X_bow, 'X_contextual': X_contextual}

        return return_dict
class PLTMDataset_tri(Dataset):

    """Class to load BoW and the contextualized embeddings for *aligned multilingual* datasets"""

    def __init__(self,X_bow1,X_bow2,X_con1,X_con2,idx2token1,idx2token2,labels=None,is_inference=False):

        self.X_bow1 = X_bow1
        self.X_bow2 = X_bow2
        self.X_con1 = X_con1
        self.X_con2 = X_con2
        self.idx2token1 = idx2token1
        self.idx2token2 = idx2token2
        self.labels = labels
        self.num_lang = 2
        self.inference_mode = is_inference
    def __len__(self):
        """Return length of dataset."""
        if self.inference_mode is False:
            return self.X_bow1[0].shape[0]
        else:
            return self.X_bow.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        # TRAINING: dataset is multilingual
        if self.inference_mode is False:
            if type(self.X_bow1[0][i]) == scipy.sparse.csr_matrix:
                X_bow1_collect = []
                X_con1_collect = []
                for l in range(self.num_lang):
                    X_bow1 = torch.FloatTensor(self.X_bow1[l][i].todense())
                    X_con1 = torch.FloatTensor(self.X_con1[l][i])
                    X_bow1_collect.append(X_bow1)
                    X_con1_collect.append(X_con1)
                X_bow1_collect = torch.stack(X_bow1_collect)
                X_con1_collect = torch.stack(X_con1_collect)
            else:
                X_bow1_collect = []
                X_con1_collect = []
                for l in range(self.num_lang):
                    X_bow1 = torch.FloatTensor(self.X_bow1[i])
                    X_con1 = torch.FloatTensor(self.X_con1[i])
                    X_bow1_collect.append(X_bow1)
                    X_con1_collect.append(X_con1)
                X_bow1_collect = torch.stack(X_bow1_collect)
                X_con1_collect = torch.stack(X_con1_collect)
            if type(self.X_bow2[0][i]) == scipy.sparse.csr_matrix:
                X_bow2_collect = []
                X_con2_collect = []
                for l in range(self.num_lang):
                    X_bow2 = torch.FloatTensor(self.X_bow2[l][i].todense())
                    X_con2 = torch.FloatTensor(self.X_con2[l][i])
                    X_bow2_collect.append(X_bow1)
                    X_con2_collect.append(X_con1)
                X_bow2_collect = torch.stack(X_bow2_collect)
                X_con2_collect = torch.stack(X_con2_collect)
            else:
                X_bow2_collect = []
                X_con2_collect = []
                for l in range(self.num_lang):
                    X_bow2 = torch.FloatTensor(self.X_bow2[i])
                    X_con2 = torch.FloatTensor(self.X_con2[i])
                    X_bow2_collect.append(X_bow2)
                    X_con2_collect.append(X_con2)
                X_bow2_collect = torch.stack(X_bow2_collect)
                X_con2_collect = torch.stack(X_con2_collect)
            return_dict = {'X_bow1': X_bow1_collect,'X_bow2': X_bow2_collect,'X_con1':X_con1_collect,'X_con2':X_con2_collect}
            # for k,v in return_dict.items():
                # print(k,v,v.shape)
        # INFERENCE: dataset is monolingual
       
        return return_dict


class PLTMDataset_tribow(Dataset):

    """Class to load BoW and the contextualized embeddings for *aligned multilingual* datasets"""

    def __init__(self,X_bow1,X_bow2,idx2token1,idx2token2,labels=None,is_inference=False):

        self.X_bow1 = X_bow1
        self.X_bow2 = X_bow2
        self.idx2token1 = idx2token1
        self.idx2token2 = idx2token2
        self.labels = labels
        self.num_lang = 2
        self.inference_mode = is_inference
    def __len__(self):
        """Return length of dataset."""
        if self.inference_mode is False:
            return self.X_bow1[0].shape[0]
        else:
            return self.X_bow.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        # TRAINING: dataset is multilingual
        if self.inference_mode is False:
            if type(self.X_bow1[0][i]) == scipy.sparse.csr_matrix:
                X_bow1_collect = []
                for l in range(self.num_lang):
                    X_bow1 = torch.FloatTensor(self.X_bow1[l][i].todense())
                    X_bow1_collect.append(X_bow1)
                X_bow1_collect = torch.stack(X_bow1_collect)
            else:
                X_bow1_collect = []
                for l in range(self.num_lang):
                    X_bow1 = torch.FloatTensor(self.X_bow1[i])
                    X_bow1_collect.append(X_bow1)
                X_bow1_collect = torch.stack(X_bow1_collect)
            if type(self.X_bow2[0][i]) == scipy.sparse.csr_matrix: 
                X_bow2_collect = []
                for l in range(self.num_lang):
                    X_bow2 = torch.FloatTensor(self.X_bow2[l][i].todense())
                    X_bow2_collect.append(X_bow2)
                X_bow2_collect = torch.stack(X_bow2_collect)
            else:
                for l in range(self.num_lang):
                    X_bow2 = torch.FloatTensor(self.X_bow2[i])
                    X_bow2_collect.append(X_bow2)
                X_bow2_collect = torch.stack(X_bow2_collect)
            return_dict = {'X_bow1': X_bow1_collect,'X_bow2': X_bow2_collect}
            # for k,v in return_dict.items():
                # print(k,v,v.shape)
        # INFERENCE: dataset is monolingual
       
        return return_dict

