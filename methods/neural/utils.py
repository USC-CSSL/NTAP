
def splitY(model, y_data, feed_dict):
    for i in range(len(model.target_cols)):
        feed_dict[model.task_outputs[model.target_cols[i]]] = y_data[:, i]
    return feed_dict