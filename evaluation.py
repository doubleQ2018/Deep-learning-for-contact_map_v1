
def accuracy(y_out, y, k):
    L = y.shape[-2]
    top_k = L/k
    y_out = y_out[:,:,:,1:]
    y = y[:,:,:,1:] ### when input shape of y is (, L, L, 2)
    y_flat = y_out.flatten()
    y = y.flatten()
    indeces = [[index, val] for index, val in enumerate(y_flat)]
    indeces = sorted(indeces, key=lambda x: x[1], reverse=True)
    right = 0
    for i in range(top_k):
        right += y[indeces[i][0]]
    return 1.0 * right / top_k


