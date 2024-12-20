vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

data_path = 'data/Ali-CCP/sample_skeleton_{}.csv'
common_feat_path = 'data/Ali-CCP/common_features_{}.csv'
enum_path = 'data/ctrcvr_enum.pkl'
write_path = 'data/ctr_cvr'
use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']

column_type = {
  'wide': ['508', '509', '702', '853'],
  'deep': ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '216', '301']
}

TRAIN_PATH = './Ali_CCP/data/ctr_cvr.train'
TRAIN_CLICKED_PATH = './Ali_CCP/data/ctr_cvr.train_clicked'
VAL_PATH = './Ali_CCP/data/ctr_cvr.dev'
TEST_PATH = './Ali_CCP/data/ctr_cvr.test'
