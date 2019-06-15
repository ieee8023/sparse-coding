import numpy as np

class Dataset:
    def __init__(self):
        pass

    def get_D(self):
        print "Must implement get_D"
        pass
    
    def get_X(self):
        print "Must implement get_X"
        pass
    
    def __str__(self):
         return str(self.__class__)


class SimpleDataset(Dataset):
    """
    Generate the specific toy dataset from the docs
    """
    def __init__(self):

        self.signals = [[1,1,3,2]]
        self.h = [[1,2]]
        
        self.A = np.asarray([[ 1.,  0.],
                             [ 1.,  1.],
                             [ 0.,  1.]])

        self.K = np.asarray([[ 1.,  0.,  0.],
                             [ 1.,  0.,  0.],
                             [ 0.,  1.,  0.],
                             [ 0.,  0.,  1.]])
        
        self.D = np.dot(self.K,self.A)
        
    def get_D(self):
        return self.D
    
    def get_signals(self):
        return self.signals
        
    def get_h(self):
        return self.h
    
class RandomDataset(Dataset):
    """
    Generate a random binary matrix D
    """
    def __init__(self, seed=0, n_in=2, n_out=4):

        # n_in  size of latent space h
        # n_out size of signal space 
        
        range_h = 10 # how high can a value in h be
        n_samples = 10
        
        np.random.seed(seed)
        self.D = np.random.choice([0.,1.],(n_out,n_in))
        self.h = np.random.choice(range(range_h),(n_samples,n_in))
    
    def get_D(self):
        return self.D
    
    def get_signals(self):
        return np.dot(self.D,self.h.T).T
        
    def get_h(self):
        return self.h
    
class NoisyRandomDataset(Dataset):
    """
    Generate a random binary matrix D
    Add gaussian noise to signals
    """
    def __init__(self, seed=0, n_in=2, n_out=4, noise=1, n_samples=10):

        # n_in  size of latent space h
        # n_out size of signal space 
        
        range_h = 10 # how high can a value in h be
        n_samples = n_samples
        
        self.noise = noise
        
        self.seed = seed
        np.random.seed(self.seed)
        self.D = np.random.choice([0.,1.],(n_out,n_in))
        self.h = np.random.choice(range(range_h),(n_samples,n_in))
    
    def get_D(self):
        return self.D
    
    def get_signals(self):
        pure = np.dot(self.D,self.h.T).T
        np.random.seed(self.seed)
        noise = np.random.normal(0, self.noise, pure.shape)
        return pure+noise
        
    def get_h(self):
        return self.h
    

class NoisySparseRandomDataset(Dataset):
    """
    Generate a random binary matrix D
    Add gaussian noise to signals
    """
    def __init__(self, seed=0, n_in=2, n_out=4, noise=1, sparsity=0.5, n_samples=10, transcript_prior="uniform"):

        # n_in  size of latent space h
        # n_out size of signal space 
        range_h = 10 # how high can a value in h be
        n_samples = n_samples
        self.noise = noise
        self.seed = seed
        
        np.random.seed(self.seed)
        #generate a transformation matrix
        self.D = np.random.choice([0.,1.], (n_out, n_in))
        
        #generate a prob for selecting a specific transcript
        if transcript_prior=="uniform":
            self.transcript_prior = [sparsity]*n_in
            
        elif transcript_prior=="fixed":
            self.transcript_prior = np.random.binomial(1, p=sparsity, size=n_in)
            print "transcript_prior=", self.transcript_prior
        else:
            raise Exception("unknown transcript_prior")
        
        np.random.seed(self.seed)
        # sample counts for all transcripts
        h = np.random.poisson(lam=range_h, size=(n_samples, n_in))
        
        # sample a mask over the counts for sparsity
        self.h = h * np.random.binomial(1, p=[self.transcript_prior]*n_samples)
 
    def get_D(self):
        return self.D

    def get_signals(self):
        pure = np.dot(self.D, self.h.T).T
        np.random.seed(self.seed)
        return abs(pure + self.noise * np.random.randn(*pure.shape))

    def get_h(self):
        return self.h
    
    
tr_cache = None
def getTRTable():
    global tr_cache
    if tr_cache is None:
        import pandas as pd
        d1 = pd.read_csv("data/table_1_11_2018.tab",delimiter="|", skiprows=[0,2], usecols=range(1, 7))
        d1.columns = d1.columns.str.strip()
        d1 = d1.iloc[:-1]

        d2 = pd.read_csv("data/table_2_11_2018.tab",delimiter="|", skiprows=[0,2], usecols=range(1, 7))
        d2.columns = d2.columns.str.strip()
        d2 = d2.iloc[:-1]

        merged = d1.merge(d2, on="u_exon_id")
        merged = merged[merged.probe_class.str.strip() != "NK0"]
        merged = merged[["u_exon_id","transcript_id","gene_id", "probe_id"]]
        tr_cache = merged.astype('int32', copy=False)

    return tr_cache
    
class MicroArrayDataset(Dataset):

    def __init__(self, seed=0, noise=0, gene_id=193174.0, n_samples=10, fake_data=True):

        self.noise = noise
        self.seed = seed
        self.gene_id = int(gene_id)
        
        self.get_D()
        
        #print self.D
        n_in = self.D.shape[1]
        
        if fake_data:
            dataset = NoisySparseRandomDataset(seed=seed, noise=noise, n_in=n_in, n_samples=n_samples, transcript_prior="fixed")
            self.h = dataset.get_h()
        

    def get_D(self, available_probes=None, possible_trancripts=None):
        
        merged = getTRTable()
        
        merged2 = merged[merged.gene_id==self.gene_id]
        
        if available_probes is not None:
            merged2 = merged2[[probe_id in available_probes for probe_id in merged2.probe_id]]
            
        if possible_trancripts is not None:
            merged2 = merged2[[transcript_id in possible_trancripts for transcript_id in merged2.transcript_id]]
        
        if merged2.shape[0] == 0:
            raise Exception("gene_id " + str(self.gene_id) + " has no exons")
        
        transcripts_raw = np.sort(merged2.transcript_id.unique())
        exons = np.sort(np.sort(merged2.u_exon_id.unique()))
        probes = np.sort(np.sort(merged2.probe_id.unique()))
        
        self.transcripts_raw = transcripts_raw
        self.exons = exons
        self.probes = probes
        
        A = []
        for exon in exons:
            transcripts_for_probes = merged2[merged2.u_exon_id == exon].transcript_id
            connected_to = [transcript in transcripts_for_probes.values for transcript in transcripts_raw]
            A.append(connected_to)
        self.A = np.asarray(A)+0.
        
        K = []
        for probe in probes:
            exons_for_probes = merged2[merged2.probe_id == probe].u_exon_id
            connected_to = [thing in exons_for_probes.values for thing in exons]
            K.append(connected_to)
        self.K = np.asarray(K)+0.
        
        self.D_raw = np.dot(self.K,self.A)
        
        
        # deduplicate transcripts
        self.D, i_unique = np.unique(self.D_raw, axis=1, return_index=True)
        self.transcripts = self.transcripts_raw[i_unique]

        #put in order
        self.D = self.D[:,np.argsort(self.transcripts)]
        self.transcripts = self.transcripts[np.argsort(self.transcripts)]
        
        
        return self.D

    def get_signals(self):
        pure = np.dot(self.D, self.h.T).T
        np.random.seed(self.seed)
        return abs(pure + self.noise * np.random.randn(*pure.shape))

    def get_h(self):
        return self.h
        
    def visualize(dataset):
        
        import matplotlib
        import matplotlib.pyplot as plt
        plt.subplot2grid((3,15), (0, 1), colspan=1)
        plt.imshow(dataset.A, cmap='winter');
        plt.yticks(np.arange(dataset.A.shape[0]), dataset.exons.astype(int));
        plt.xticks(np.arange(dataset.A.shape[1]), dataset.transcripts_raw.astype(int), rotation='vertical');
        plt.ylabel("Exons");
        plt.xlabel("Transcripts");
        plt.clim(0,1) # make 1s are always the same color
        plt.grid()
        plt.title("A");

        plt.subplot2grid((3,15), (0, 2), colspan=14)
        plt.imshow(dataset.K.T, cmap='winter');
        plt.yticks(np.arange(dataset.K.T.shape[0]), [""]*dataset.K.T.shape[0]);
        plt.xticks(np.arange(dataset.K.T.shape[1]), dataset.probes.astype(int), rotation='vertical');
        plt.xlabel("Probes");
        plt.clim(0,1)
        plt.grid()
        plt.title("Gene: " + str(int(dataset.gene_id)) + "\n K.T")

        plt.subplot2grid((3,15), (1, 1), colspan=15)
        plt.imshow(dataset.D_raw.T, cmap='winter');
        plt.yticks(np.arange(dataset.D_raw.T.shape[0]), dataset.transcripts_raw.astype(int));
        plt.xticks(np.arange(dataset.D_raw.T.shape[1]), dataset.probes.astype(int), rotation='vertical');
        plt.ylabel("Transcripts");
        plt.xlabel("Probes");
        plt.clim(0,1)
        plt.grid();
        plt.title("D_raw.T");
        
        plt.subplot2grid((3,15), (2, 1), colspan=15)
        plt.imshow(dataset.D.T, cmap='winter');
        plt.yticks(np.arange(dataset.D.T.shape[0]), dataset.transcripts.astype(int));
        plt.xticks(np.arange(dataset.D.T.shape[1]), dataset.probes.astype(int), rotation='vertical');
        plt.ylabel("Transcripts");
        plt.xlabel("Probes");
        plt.clim(0,1);
        plt.grid();
        plt.title("D.T");
        
        plt.show()


