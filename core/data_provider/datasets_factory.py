#数据处理工厂-数据粗处理
from core.data_provider import  mnist
datasets_map = {
    'mnist': mnist,
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, injection_action, is_training=True):
    if dataset_name not in datasets_map:#异常处理
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')#训练数据list=训练数据路径，用','隔开
    valid_data_list = valid_data_paths.split(',')#验证数据list=验证数据路径，用','隔开
    if dataset_name == 'mnist':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'} #测试输入的参数配置test_input_param
        # test_input_handle为mnist.py中的class InputHandle类的对象
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False) #begin()为InputHandle类中的方法，不打乱，作用：？？？
        if is_training: #需要训练过程时
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}#训练输入的参数配置train_input_param
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param) #训练输入的对象
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle



