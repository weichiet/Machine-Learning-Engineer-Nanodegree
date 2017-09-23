
import tensorflow as tf
#%%

def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  """
  Compute the confusion metrics of the prediction results
  This function is based on the code published on https://goo.gl/uh3hws
  """  
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op], 
      feed_dict
    )

  tpfn = float(tp) + float(fn)
  tpr = 0 if tpfn == 0 else float(tp)/tpfn
  fpr = 0 if tpfn == 0 else float(fp)/tpfn

  total = float(tp) + float(fp) + float(fn) + float(tn)
  accuracy = 0 if total == 0 else (float(tp) + float(tn))/total

  recall = tpr
  tpfp = float(tp) + float(fp)
  precision = 0 if tpfp == 0 else float(tp)/tpfp
  
  f1_score = 0 if recall == 0 else (2 * (precision * recall)) / (precision + recall)
  
  #print('Precision = {:.4f}'.format(precision))
  #print('Recall = {:.4f}'.format(recall))
  print('F1 Score = {:.4f}'.format(f1_score))
  print('Accuracy = {:.4f}'.format(accuracy))

  return accuracy, f1_score
#%%
# Create model
def multilayer_perceptron_default(x, weights, biases):
    """Create tensorflow model with specified input, weights and biases"""
    
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
       
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    #model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)
    
    return out_layer

#%%

def train_mlp_default(X_train, y_train, X_test, y_test,training_epochs=2000):
    """Train a 2-layers MLP using tensorflow library"""
    
    # Parameters
    learning_rate = 0.0001
  
     # 'Saver' op to save and restore all the variables
    #saver = tf.train.Saver()
    #model_path = "tmp/model.ckpt"   
    
    # Network Parameters
    n_hidden_1 = 25 # 1st layer number of features
    n_hidden_2 = 50 # 2nd layer number of features
    n_input = X_train.shape[1] # Number of input features
    n_classes = y_train.shape[1] # Number of output
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
        
     # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.0001)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.0001)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.0001))
    }
    
    biases = {
        'b1': tf.Variable(tf.ones([n_hidden_1])),
        'b2': tf.Variable(tf.ones([n_hidden_2])),
        'out': tf.Variable(tf.ones([n_classes]))
    }
    # Construct model
    model = multilayer_perceptron_default(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.global_variables_initializer()   
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
    
        # Training cycle
        for epoch in range(training_epochs):
            sess.run([optimizer, cost], feed_dict={x: X_train,
                                                     y: y_train})    
            
            if epoch%500== 0:			
                print("Epoch:",'%d' %(epoch), "Accuracy=", \
                    "{:.4f}".format(sess.run(accuracy,
                              feed_dict={
                                x: X_train, 
                                y: y_train
                              }
                            )))
                                                          
        print ("MLP Training Finished!")
    
        # Save model weights to disk
        #save_path = saver.save(sess, model_path)
        #print ("Model saved in file: %s\n" % save_path)
        
        feed_dict= {
          x: X_test,
          y: y_test
        }
        
        accuracy, f1_score = tf_confusion_metrics(model, y, sess, feed_dict)
    
    return accuracy, f1_score
#%%

def multilayer_perceptron_tuned(x, weights, biases):
    """Create tensorflow model with specified input, weights and biases"""
    
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)    
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)    

    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
   
    return out_layer

#%%

def train_mlp_tuned(X_train, y_train, X_test, y_test,training_epochs=1500):
    """Train a 3-layers MLP using tensorflow library"""
    
    # Parameters
    learning_rate = 0.0001
  
     # 'Saver' op to save and restore all the variables
    #saver = tf.train.Saver()
    #model_path = "tmp/model.ckpt"   
    
    # Network Parameters
    n_hidden_1 = 50 # 1st layer number of features
    n_hidden_2 = 100 # 2nd layer number of features
    n_hidden_3 = 150 # 3nd layer number of features    
    n_input = X_train.shape[1] # Number of input features
    n_classes = y_train.shape[1] # Number of output
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
        
     # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.0001)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.0001)),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.0001)),        
        'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_classes], stddev=0.0001))
    }
    
    biases = {
        'b1': tf.Variable(tf.ones([n_hidden_1])),
        'b2': tf.Variable(tf.ones([n_hidden_2])),
        'b3': tf.Variable(tf.ones([n_hidden_3])),           
        'out': tf.Variable(tf.ones([n_classes]))
    }
    # Construct model
    model = multilayer_perceptron_tuned(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.global_variables_initializer()   
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
    
        # Training cycle
        for epoch in range(training_epochs):
            sess.run([optimizer, cost], feed_dict={x: X_train,
                                                     y: y_train})    
            
            if epoch%500== 0:			
                print("Epoch:",'%d' %(epoch), "Accuracy=", \
                    "{:.4f}".format(sess.run(accuracy,
                              feed_dict={
                                x: X_train, 
                                y: y_train
                              }
                            )))
                                                          
        print ("MLP Training Finished!")
    
        # Save model weights to disk
        #save_path = saver.save(sess, model_path)
        #print ("Model saved in file: %s\n" % save_path)
        
        feed_dict= {
          x: X_test,
          y: y_test
        }
        
        accuracy, f1_score = tf_confusion_metrics(model, y, sess, feed_dict)
    
    return accuracy, f1_score
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    