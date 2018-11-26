__author__ = '15275'

import tensorflow as tf
import pandas as pd

graph1 = tf.Graph()
with graph1.as_default():
    class deepFMmodel():
        def __init__(self,args):
            #存放每deep部分没层神经元个数
            self.layers=args['layers']
            self.embeding_size=args['embeding_size'];
            self.weight={};
            self.feature_size=args['feature_size']
            self.field_size=args['field_size']
            self.last_layer={}
            self.ac_function=args['ac_function']
            self.learing_rate=args['learning_rate']
            self.save_path=args['save_path']
            self.build_model();
        def build_model(self):
            self.feature_index=tf.placeholder(dtype=tf.int32,shape=[None,None])
            self.feature_value=tf.placeholder(dtype=tf.float32,shape=[None,None])
            self.y_true=tf.placeholder(dtype=tf.float32,shape=[None,None])
            #声明 embeding层参数  这个weight其实就是v

            self.weight['embeding_weight']=tf.Variable(initial_value=tf.truncated_normal(shape=(self.feature_size,self.embeding_size),mean=0.0,stddev=0.1,dtype=tf.float32),dtype=tf.float32)
            #声明一阶关系参数
            self.weight['one_order_weight']=tf.Variable(initial_value=tf.truncated_normal(shape=(self.feature_size,1),mean=0.0,stddev=0.1,dtype=tf.float32),dtype=tf.float32)
            self.weight['one_order_baise']=tf.Variable(initial_value=tf.zeros(shape=(1,1)),dtype=tf.float32)
            #二阶关系参数就是一阶关系参数
            #deep参数
            deep_input_size=self.field_size*self.embeding_size;
            self.weight['deep_weight_0']=tf.Variable(initial_value=tf.truncated_normal(shape=(deep_input_size,self.layers[0]),mean=0.0,stddev=0.1,dtype=tf.float32),dtype=tf.float32);
            self.weight['deep_baise_0']=tf.Variable(initial_value=tf.zeros(shape=(1,self.layers[0]),dtype=tf.float32),dtype=tf.float32)
            for i in range(len(self.layers)):
                if i>0:
                    self.weight['deep_weight_'+str(i)]=tf.Variable(initial_value=tf.truncated_normal(shape=(self.layers[i-1],self.layers[i]),mean=0.0,stddev=0.1,dtype=tf.float32),dtype=tf.float32);
                    self.weight['deep_baise_'+str(i)]=tf.Variable(initial_value=tf.zeros(shape=(1,self.layers[i]),dtype=tf.float32),dtype=tf.float32)
            #声明最后一层参数
            last_layer_input_size=1+self.layers[-1]+self.embeding_size
            self.last_layer['weight']=tf.Variable(initial_value=tf.truncated_normal(shape=(last_layer_input_size,8),mean=0.0,stddev=0.1,dtype=tf.float32),dtype=tf.float32);
            self.last_layer['baise']=tf.Variable(initial_value=tf.zeros(shape=(1,8),dtype=tf.float32),dtype=tf.float32);
            self.last_layer['weight2']=tf.Variable(initial_value=tf.truncated_normal(shape=(8,1),mean=0.0,stddev=0.1,dtype=tf.float32),dtype=tf.float32);
            self.last_layer['baise2']=tf.Variable(initial_value=tf.zeros(shape=(1,1),dtype=tf.float32),dtype=tf.float32);
            #参数声明完毕  构建网络
            #embeding层构建
            embeding_matrix=tf.nn.embedding_lookup(self.weight['embeding_weight'],self.feature_index);
            embeding_output=tf.multiply(embeding_matrix,tf.reshape(self.feature_value,shape=[-1,self.field_size,1]))
            #一阶关系构建
            one_order_matrix=tf.nn.embedding_lookup(self.weight['one_order_weight'],self.feature_index);
            #print('feature_value:',self.feature_value)
            one_order_layer=tf.multiply(one_order_matrix,tf.reshape(self.feature_value,shape=[-1,self.field_size,1]))
            one_order_layer_output=tf.reduce_sum(one_order_layer,1);

            print("one_order_layer_output:",one_order_layer_output)
            #二阶关系
            #计算squre(∑vif*xi)
            mul_sum=tf.reduce_sum(embeding_output,1)
            mul_sum_squre=tf.square(mul_sum);
            #计算∑squre(vif*xi)
            mul_squre=tf.square(embeding_output)
            mul_squre_sum=tf.reduce_mean(mul_squre,1);
            two_order_output=0.5*tf.subtract(mul_sum_squre,mul_squre_sum)
            fm_output=tf.concat([one_order_layer_output,two_order_output],axis=1);
            #deep阶段
            deep_layer=tf.reshape(embeding_output,[-1,deep_input_size])
            print("deep_layer:",deep_layer)
            for i in range(len(self.layers)):
                if i>0 or i==0:
                    deep_layer=tf.matmul(deep_layer,self.weight["deep_weight_%d"%i]);
                    deep_layer=self.ac_function(tf.add(deep_layer,self.weight["deep_baise_%d"%i]));
            deep_output=tf.concat([fm_output,deep_layer],axis=1);
            last1=self.ac_function(tf.add(tf.matmul(deep_output,self.last_layer['weight']),self.last_layer['baise']));
            self.predict=tf.add(tf.matmul(last1,self.last_layer['weight2']),self.last_layer['baise2']);
            #predict=tf.add(tf.matmul(last2,self.last_layer['weight3']),self.last_layer['baise3']);
            self.loss=tf.reduce_mean(tf.square(self.y_true-self.predict));
            # for n in range(len(self.layers)):
            #     self.loss+=tf.contrib.layers.l2_regularizer(0.1)(self.weight['deep_weight_%d'%i]);
            # self.loss+=tf.contrib.l2_regularizer(0.1)(self.last_layer['weight'])
            optimizer=tf.train.GradientDescentOptimizer(self.learing_rate);
            self.train_step=optimizer.minimize(self.loss);
        def train(self,session,index,value,label):
            loss,_=session.run([self.loss,self.train_step],feed_dict={self.feature_index:index,self.feature_value:value,self.y_true:label})
            return loss;
        def get_global_loss(self,session,index,value,label):
            global_loss=session.run([self.loss],feed_dict={self.feature_index:index,self.feature_value:value,self.y_true:label})
            return global_loss;
        def prediction(self,session,index,value,label):
            predict=session.run([self.predict],feed_dict={self.feature_index:index,self.feature_value:value,self.y_true:label})
            return predict;
        def save(self,session):
            saver=tf.train.Saver();
            saver.save(session,save_path=self.save_path)
        def restore(self,session):
            saver=tf.train.Saver()
            saver.restore(session,save_path=self.save_path)
        def gene_batch(self,x_value,x_index,label,batch_size):
            sum_number=len(x_value)
            batch_number=sum_number//batch_size;
            for i in range(batch_number):
                m=i+1;

                yield x_value[(m-1)*batch_size:m*batch_size],x_index[(m-1)*batch_size:m*batch_size],label[(m-1)*batch_size:m*batch_size];

    def load_data():
        train_data = {}
        file_path = 'D:/study/论文数据/train_no_latend.csv'
        label_path='D:/study/论文数据/train_rating_label.csv'
        label=pd.read_csv(label_path,header=None)
        print(label.shape)
        data = pd.read_csv(file_path).iloc[:,0:-1]
        print(data.shape)
        data_label=pd.concat([data,label],axis=1);

        data=data_label.iloc[:,0:8]
        label=data_label.iloc[:,-1]
        label=pd.Series.reshape(label,[-1,1])

        data.columns = ['c' + str(i) for i in range(data.shape[1])]

        train_data['y_train'] = label
        co_feature = pd.DataFrame()
        ca_feature = pd.DataFrame()
        ca_col = []
        co_col = []
        feat_dict = {}
        cnt = 1
        for i in range(0, data.shape[1]):
            target = data.iloc[:, i]
            col = target.name
            l = len(set(target))
            if l > 0:
                target = (target - target.mean()) / target.std()
                co_feature = pd.concat([co_feature, target], axis=1)
                feat_dict[col] = cnt
                cnt += 1
                co_col.append(col)
            else:
                #去重
                us = target.unique()
                feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))
                ca_feature = pd.concat([ca_feature, target], axis=1)
                cnt += len(us)
                ca_col.append(col)
        feat_dim = cnt
        feature_value_1 = pd.concat([co_feature, ca_feature], axis=1)
        feature_index_1 = feature_value_1.copy()
        for i in feature_index_1.columns:
            if i in co_col:
                feature_index_1[i] = feat_dict[i]
            else:
                feature_index_1[i] = feature_index_1[i].map(feat_dict[i])
                feature_value_1[i] = 1.
        train_data['xi'] = feature_index_1.values.tolist()
        train_data['xv'] = feature_value_1.values.tolist()
        train_data['feat_dim'] = feat_dim
        return train_data

    data=load_data();
    feature_size=data['feat_dim']
    field_size=len(data['xi'][0])
    args={
        'layers':[64,32,16],
        'embeding_size':128,
        'feature_size':feature_size,
        'field_size':field_size,
        'ac_function':tf.nn.relu,
        'save_path':'D:/study/model/deepfm',
        'learning_rate':0.05,
        'epoch':6000
    }
    session=tf.Session(graph=graph1);
    my_model=deepFMmodel(args);
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    loss_list=[];
    for i in range(args['epoch']):
        for x_value,x_index,label in my_model.gene_batch(data['xv'],data['xi'],data['y_train'],1000):
            loss_batch=my_model.train(session,x_index,x_value,label)
        if  i%100==0:
            loss_all=my_model.get_global_loss(session,data['xi'],data['xv'],data['y_train'])
            print("loss_all",loss_all)
            my_model.save(session)
            loss_list.append(loss_all)
    loss_df=pd.DataFrame(loss_list);
    loss_df.to_csv("D:/study/论文数据/loss/deepfm/deepfm_nolanten_loss.csv")


