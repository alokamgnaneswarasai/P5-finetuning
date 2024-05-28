"""
    Running MF / SVD Recommenders [Rating Prediction]

    - Cross Validation
    - Simple

"""

from caserecc.recommenders.rating_prediction.svdplusplus import SVDPlusPlus
from caserecc.recommenders.rating_prediction.nnmf import NNMF
from caserecc.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserecc.utils.cross_validation import CrossValidation

db = '../../datasets/ml-100k/u.data'
folds_path = '../../datasets/ml-100k/'

metadata_item = '../../datasets/ml-100k/db_item_subject.dat'
sm_item = '../../datasets/ml-100k/sim_item.dat'
metadata_user = '../../datasets/ml-100k/metadata_user.dat'
sm_user = '../../datasets/ml-100k/sim_user.dat'

tr = '../../datasets/ml-100k/folds/0/train.dat'
te = '../../datasets/ml-100k/folds/0/test.dat'

"""

    UserKNN

"""

# Cross Validation
# recommender = MatrixFactorization()

# CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

# # Simple
# MatrixFactorization(tr, te).compute()
# SVDPlusPlus(tr, te).compute()

NNMF(tr, te, factors = 20).compute()