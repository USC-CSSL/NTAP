"""
define abstract class `Model`
Methods:
    CV
    set_params
    train
    summary

"""



    def predict(self, data, verbose='minimal'): # weights?
        """
        return pandas array of predictions
        """
        acc_test = 0
        test_predictions = {target: np.array([]) for target in self.targets}
        test_labels = {target: np.array([]) for target in self.targets}
        for feed_dict in self.batches(data):
            acc_test += self.joint_accuracy.eval(feed_dict=feed_dict)
            for i in range(len(self.target_cols)):
                test_predictions[self.target_cols[i]] = np.append(test_predictions[self.target_cols[i]],
                                                                   self.predict[self.target_cols[i]].eval(
                                                                       feed_dict=feed_dict))
                test_labels[self.target_cols[i]] = np.append(test_labels[self.target_cols[i]],
                                                              feed_dict[
                                                                  self.task_outputs[self.target_cols[i]]])
        return acc_test, test_predictions, test_labels

    @abstractmethod
    def summary(...):
        print("Performance")

    @abstractmethod
    def __str__(self):


    def buildEmbedding(self, pretrain, train_embedding, embedding_size, vocab_size, expand_dims): # expand_dims??
        if pretrain:
            embeddings_variable = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=train_embedding, name="W")
        else:
            embeddings_variable = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [vocab_size, embedding_size], -1, 1),
                                                    dtype=tf.float32)
        if expand_dims==True:
            embeddings_variable = tf.expand_dims(embeddings_variable,-1)  # ??
        return embeddings_variable

class RNN(Model):
    def __init__(self, formula, ...):
        self.__parse_formula(formula)
        ...
    def __parse_formula(self, formula)
        # formula: "hate ~ text"
        targets, inputs = formula.split("~")
        self.targets = targets.split(); self.inputs = inputs.split()
        #self.text_col = inputs[0]
        #if self.
        # later: parse `ddr(text)` and similar. Can specify dictionary=`./mfd.json`)

    ...

"""
    def buildPredictor(self):
        self.loss, self.accuracy, self.predict = dict(), dict(), dict()

        for target in self.targets:
            self.loss[target], self.predict[target], self.accuracy[target] = self.get_accuracy_loss_predictedLabel(self.state, self.n_outputs, self.weights[target], self.task_outputs[target])
        self.joint_accuracy = sum(self.accuracy.values()) / len(self.target_cols)
        self.joint_loss = sum(self.loss.values())
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)

    # a function to build a single RNN cell with Dropout
    def buildSingle_RNN(self, cell, hidden_size, keep_ratio):
        if cell == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, reuse=tf.AUTO_REUSE)
        elif cell == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_ratio)
        return cell_drop

    # a function to construct dynamic_rnn or bidirectional_dynamic_rnn model
    def dynamic_rnn(self, cell, model, hidden_layers, keep_prob, embed, sequence_length):
        if model[:4] == "LSTM":
            network = self.multi_RNN(cell, hidden_layers, keep_prob)
            rnn_outputs, state = tf.nn.dynamic_rnn(network, embed, dtype=tf.float32, sequence_length=sequence_length)
            if cell == "GRU":
                state = state[0]
            else:
                state = state[0].h
        else:
            f_network = self.multi_RNN(cell, hidden_layers, keep_prob)
            b_network = self.multi_RNN(cell, hidden_layers, keep_prob)
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(f_network, b_network, embed, dtype=tf.float32, sequence_length=sequence_length)
            fw_outputs, bw_outputs = bi_outputs
            fw_states, bw_states = bi_states
            rnn_outputs = tf.concat([fw_outputs, bw_outputs], 2)
            if cell == "GRU":
                state = tf.concat([fw_states[0], bw_states[0]], 1)
            else:
                state = tf.concat([fw_states[0].h, bw_states[0].h], 1)
        return rnn_outputs, state

    # a function to provide input values to the model variables
    def feed_dictionary(self, batch, weights):
        feed_dict = {self.train_inputs: batch["text"],
                    self.sequence_length: batch["sent_lens"],
                    self.keep_prob: self.keep_ratio}
        if len(batch["label"]) > 0:
            feed_dict = self.splitY(batch["label"], feed_dict)
        else:
            feed_dict[self.keep_prob] = 1

        if self.feature:
            feed_dict[self.features] = batch["feature"]

        for t in self.target_cols:
            feed_dict[self.weights[t]] = weights[t]
        if self.pretrain:
            feed_dict[self.embedding_placeholder] = self.my_embeddings
        return feed_dict

    # a function to calculate the precision, recall and f1 scores based on the true labels and predicted labels
    def get_precision_recall_f1_scores(self, test_predictions, test_labels):
        f1_scores = dict()
        precisions = dict()
        recalls = dict()
        for i in range(len(self.target_cols)):
            score = f1_score(test_predictions[self.target_cols[i]],
                             test_labels[self.target_cols[i]],
                             average = "macro" if self.n_outputs > 2 else "binary")
            pres = precision_score(test_predictions[self.target_cols[i]],
                             test_labels[self.target_cols[i]],
                             average = "macro" if self.n_outputs > 2 else "binary")
            rec = recall_score(test_predictions[self.target_cols[i]],
                             test_labels[self.target_cols[i]],
                             average = "macro" if self.n_outputs > 2 else "binary")
            print("F1", self.target_cols[i], score,
                  "Precision", self.target_cols[i], pres,
                  "Recall", self.target_cols[i], rec)
            f1_scores[self.target_cols[i]] = score
            precisions[self.target_cols[i]] = pres
            recalls[self.target_cols[i]] = rec
        return f1_scores, precisions, recalls

    # a function to initialise variables
    def initialise(self):
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = self.buildEmbedding(self.pretrain, self.train_embedding, self.embedding_size, len(self.vocab), self.expand_dims)
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.task_outputs = self.multi_outputs(self.target_cols)
        self.weights = self.weightPlaceholder(self.target_cols)
        self.keep_prob = tf.placeholder(tf.float32)



    # a function to build multiple RNN cells
    def multi_RNN(self, cell, hidden_layers, keep_ratio):
        network = tf.contrib.rnn.MultiRNNCell([self.buildSingle_RNN(cell, hidden_size, keep_ratio) for hidden_size in hidden_layers])
        return network

    # a function to define a variable for every target in case of multiple targets
    def multi_outputs(self,target_cols):
        outputs = dict()
        for target in target_cols:
            y = tf.placeholder(tf.int64, [None], name=target)
            outputs[target] = y
        return outputs

    # a function to calculate accuracy, loss and predicted label for a single target
    def get_accuracy_loss_predictedLabel(self, hidden, n_outputs, weights, task_outputs):
        logits = tf.layers.dense(hidden, n_outputs)

        weight = tf.gather(weights, task_outputs)
        xentropy = tf.losses.sparse_softmax_cross_entropy(labels=task_outputs,
                                                          logits=logits,
                                                          weights=weight)
        loss = tf.reduce_mean(xentropy)
        predicted_label = tf.argmax(logits, 1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_label, task_outputs), tf.float32))

        return loss, predicted_label, accuracy

    # a function to predict the label on the trained model
    def predictModel(self, batches, data_batches, weights, savedir):
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            try:
                saver.restore(self.sess, getModelDirectory(self.all_params)+"/finalModel")
            except Exception as e:
                print(e)
                print("No saved model. Train a model before prediction")
                exit(1)

            label_predictions = {target: np.array([]) for target in self.target_cols}
            for i in range(len(data_batches)):
                feed_dict = self.feed_dictionary(data_batches[i], weights)
                for j in range(len(self.target_cols)):
                    label_predictions[self.target_cols[j]] = np.append(label_predictions[self.target_cols[j]],
                                                                       self.predict[self.target_cols[j]].eval(
                                                                           feed_dict=feed_dict))
                if i % 1000 == 0 and i > 0:
                    print(i)
                    results = pd.DataFrame.from_dict(label_predictions)
                    results.to_csv(savedir + "/predictions_" + str(i * self.batch_size) + ".csv")
                    label_predictions = {target: np.array([]) for target in self.target_cols}
            results = pd.DataFrame.from_dict(label_predictions)
            results.to_csv(savedir + "/predictions_" + str(i * self.batch_size) + ".csv")

    def splitY(self, y_data, feed_dict):
        for i in range(len(self.target_cols)):
            feed_dict[self.task_outputs[self.target_cols[i]]] = y_data[:, i]
        return feed_dict

    # a function to train a model
    def trainModel(self, batches, test_batches, weights):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        patience = 3
        model_path = getModelDirectory(self.all_params)
        monitor_test_acc, monitor_test_predictions, monitor_test_labels = [], [], []
        with tf.Session() as self.sess:
            done = False
            init.run()
            epoch = 1
            while epoch<=self.epochs:
                final_epoch = epoch
                ## Train
                acc_train, epoch_loss = self.model_training(batches, weights)
                ## Test
                acc_test, test_predictions, test_labels = self.model_testing(test_batches, weights)
                monitor_test_acc.append(acc_test)
                monitor_test_predictions.append(test_predictions)
                monitor_test_labels.append(test_labels)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Train Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", acc_test / float(len(test_batches)))
                # Early Stopping
                if epoch==1:
                    final_acc = acc_test
                    print("Saving the model at epoch:", epoch)
                    saver.save(self.sess, model_path+"/model")
                    i = 1
                else:
                    if final_acc>=acc_test:
                        i+=1
                        if i>patience:
                            final_epoch = epoch-patience
                            print("Invoking Early Stopping")
                            print("Final model stored after epoch: "+str(final_epoch))
                            break

                    else:
                        i = 1
                        final_acc = acc_test
                        print("Saving the model at epoch:", epoch)
                        saver.save(self.sess, model_path+"/model")
                epoch += 1

            f1_scores, precisions, recalls = self.get_precision_recall_f1_scores(monitor_test_predictions[final_epoch-1], monitor_test_labels[final_epoch-1])

        return f1_scores, precisions, recalls

    # a function to define a weight variable corresponding to every target in case of multiple targets
    def weightPlaceholder(self, target_cols):
        weights = dict()
        for target in target_cols:
            weights[target] = tf.placeholder(tf.float64, [None], name=target + "_w")
        return weights

class CNN(Model):
    def __init__(self, formula, ...):
        self.__parse_formula(formula)
        ...
    def __parse_formula(self, formula)
        ...

    def build_CNN(self, input, filter_sizes, num_filters, keep_ratio):
        pooled_outputs = list()
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, int(input.get_shape()[2]), 1, num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))

            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        output = tf.nn.dropout(h_pool_flat, keep_ratio)
        return output
    ...

class SVM(Model):
    def __init__(self, formula, ...):
        self.__parse_formula(formula)
        ...
    def __parse_formula(self, formula)
        ...
class LogReg(Model):
    def __init__(self, formula, ...):
        self.__parse_formula(formula)
        ...
    def __parse_formula(self, formula)
        ...

class Regression(Model):
    def __init__(self, formula, ...):
        self.__parse_formula(formula)
        ...
    def __parse_formula(self, formula)
        ...
