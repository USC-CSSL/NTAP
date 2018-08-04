
    if feature_reduce and feature_reduce != 0:
        X, lookup_dict = reduce_features(X, feature_reduce, lookup_dict)
        # rearrange lookup dict so that the keys include all values between 0 to len(features)
        lookup_dict = {i: feat for i, feat in enumerate(lookup_dict.values())}



def reduce_features(X, feature_reduce, lookup):
    drop_columns = []
    for col in range(X.shape[1]):
        feat, freq = np.unique(X[:,col], return_counts=True)
        if float(max(freq)) / float(X.shape[0]) > 1 - feature_reduce:
            drop_columns.append(col)

    for col in drop_columns:
        print("Dropping columns " + lookup[col])
        lookup.pop(col)
    if type(X) == np.ndarray:
        X = np.delete(X, drop_columns, 1)
    #X1 = sparse.csr_matrix(np.array(X))
    else:
        X = dropcols_coo(X, drop_columns)
    return X, lookup

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()
