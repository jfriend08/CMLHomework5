class MiniBatchKMeans(KMeans):
    """Mini-Batch K-Means clustering
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, optional
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.
    tol : float, default: 0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.
        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).
    batch_size : int, optional, default: 100
        Size of the mini batches.
    init_size : int, optional, default: 3 * batch_size
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.
    init : {'k-means++', 'random' or an ndarray}, default: 'k-means++'
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.
    compute_labels : boolean, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    reassignment_ratio : float, default: 0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : boolean, optional
        Verbosity mode.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point (if compute_labels is set to True).
    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition (if compute_labels is set to True). The inertia is
        defined as the sum of square distances of samples to their nearest
        neighbor.
    Notes
    -----
    See http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
    """

    def __init__(self, n_clusters=8, init='k-means++', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01):

        super(MiniBatchKMeans, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            verbose=verbose, random_state=random_state, tol=tol, n_init=n_init)

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    def fit(self, X, y=None):
        """Compute the centroids on X by chunking it into mini-batches.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Coordinates of the data points to cluster
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse="csr", order='C', dtype=np.float64)
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples smaller than number "
                             "of clusters.")

        n_init = self.n_init
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=np.float64)
            if n_init != 1:
                warnings.warn(
                    'Explicit initial center position passed: '
                    'performing only one init in MiniBatchKMeans instead of '
                    'n_init=%d'
                    % self.n_init, RuntimeWarning, stacklevel=2)
                n_init = 1

        x_squared_norms = row_norms(X, squared=True)

        if self.tol > 0.0:
            tol = _tolerance(X, self.tol)

            # using tol-based early stopping needs the allocation of a
            # dedicated before which can be expensive for high dim data:
            # hence we allocate it outside of the main loop
            old_center_buffer = np.zeros(n_features, np.double)
        else:
            tol = 0.0
            # no need for the center buffer if tol-based early stopping is
            # disabled
            old_center_buffer = np.zeros(0, np.double)

        distances = np.zeros(self.batch_size, dtype=np.float64)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)

        init_size = self.init_size
        if init_size is None:
            init_size = 3 * self.batch_size
        if init_size > n_samples:
            init_size = n_samples
        self.init_size_ = init_size

        validation_indices = random_state.random_integers(
            0, n_samples - 1, init_size)
        X_valid = X[validation_indices]
        x_squared_norms_valid = x_squared_norms[validation_indices]

        # perform several inits with random sub-sets
        best_inertia = None
        for init_idx in range(n_init):
            if self.verbose:
                print("Init %d/%d with method: %s"
                      % (init_idx + 1, n_init, self.init))
            counts = np.zeros(self.n_clusters, dtype=np.int32)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size)

            # Compute the label assignment on the init dataset
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X_valid, x_squared_norms[validation_indices],
                cluster_centers, counts, old_center_buffer, False,
                distances=None, verbose=self.verbose)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, x_squared_norms_valid,
                                         cluster_centers)
            if self.verbose:
                print("Inertia for init %d/%d: %f"
                      % (init_idx + 1, n_init, inertia))
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = counts
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.random_integers(
                0, n_samples - 1, self.batch_size)

            # Perform the actual update step on the minibatch data
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + self.counts_.min()) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        self.n_iter_ = iteration_idx + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = self._labels_inertia_minibatch(X)

        return self

    def _labels_inertia_minibatch(self, X):
        """Compute labels and inertia using mini batches.
        This is slightly slower than doing everything at once but preventes
        memory errors / segfaults.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        labels : array, shap (n_samples,)
            Cluster labels for each point.
        inertia : float
            Sum of squared distances of points to nearest cluster.
        """
        if self.verbose:
            print('Computing label assignment and total inertia')
        x_squared_norms = row_norms(X, squared=True)
        slices = gen_batches(X.shape[0], self.batch_size)
        results = [_labels_inertia(X[s], x_squared_norms[s],
                                   self.cluster_centers_) for s in slices]
        labels, inertia = zip(*results)
        return np.hstack(labels), np.sum(inertia)

    def partial_fit(self, X, y=None):
        """Update k means estimate on a single mini-batch X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Coordinates of the data points to cluster.
        """

        X = check_array(X, accept_sparse="csr")
        n_samples, n_features = X.shape
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=np.float64)

        if n_samples == 0:
            return self

        x_squared_norms = row_norms(X, squared=True)
        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))
        if (not hasattr(self, 'counts_')
                or not hasattr(self, 'cluster_centers_')):
            # this is the first call partial_fit on this object:
            # initialize the cluster centers
            self.cluster_centers_ = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=self.random_state_,
                x_squared_norms=x_squared_norms, init_size=self.init_size)

            self.counts_ = np.zeros(self.n_clusters, dtype=np.int32)
            random_reassign = False
            distances = None
        else:
            # The lower the minimum count is, the more we do random
            # reassignment, however, we don't want to do random
            # reassignment too often, to allow for building up counts
            random_reassign = self.random_state_.randint(
                10 * (1 + self.counts_.min())) == 0
            distances = np.zeros(X.shape[0], dtype=np.float64)

        _mini_batch_step(X, x_squared_norms, self.cluster_centers_,
                         self.counts_, np.zeros(0, np.double), 0,
                         random_reassign=random_reassign, distances=distances,
                         random_state=self.random_state_,
                         reassignment_ratio=self.reassignment_ratio,
                         verbose=self.verbose)

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia(
                X, x_squared_norms, self.cluster_centers_)

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return self._labels_inertia_minibatch(X)[0]



def _mini_batch_step(X, x_squared_norms, centers, counts,
                     old_center_buffer, compute_squared_diff,
                     distances, random_reassign=False,
                     random_state=None, reassignment_ratio=.01,
                     verbose=False):
    """Incremental update of the centers for the Minibatch K-Means algorithm.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The original data array.
    x_squared_norms : array, shape (n_samples,)
        Squared euclidean norm of each data point.
    centers : array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE
    counts : array, shape (k,)
         The vector in which we keep track of the numbers of elements in a
         cluster. This array is MODIFIED IN PLACE
    distances : array, dtype float64, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to its closest center.
        May not be None when random_reassign is True.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    random_reassign : boolean, optional
        If True, centers with very low counts are randomly reassigned
        to observations.
    reassignment_ratio : float, optional
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : bool, optional, default False
        Controls the verbosity.
    compute_squared_diff : bool
        If set to False, the squared diff computation is skipped.
    old_center_buffer : int
        Copy of old centers for monitoring convergence.
    Returns
    -------
    inertia : float
        Sum of distances of samples to their closest cluster center.
    squared_diff : numpy array, shape (n_clusters,)
        Squared distances between previous and updated cluster centers.
    """
    # Perform label assignment to nearest centers
    nearest_center, inertia = _labels_inertia(X, x_squared_norms, centers,
                                              distances=distances)

    if random_reassign and reassignment_ratio > 0:
        random_state = check_random_state(random_state)
        # Reassign clusters that have very low counts
        to_reassign = counts < reassignment_ratio * counts.max()
        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > .5 * X.shape[0]:
            indices_dont_reassign = np.argsort(counts)[int(.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = choice(X.shape[0], replace=False, size=n_reassigns,
                                 random_state=random_state)
            if verbose:
                print("[MiniBatchKMeans] Reassigning %i cluster centers."
                      % n_reassigns)

            if sp.issparse(X) and not sp.issparse(centers):
                assign_rows_csr(X,
                                astype(new_centers, np.intp),
                                astype(np.where(to_reassign)[0], np.intp),
                                centers)
            else:
                centers[to_reassign] = X[new_centers]
        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        counts[to_reassign] = np.min(counts[~to_reassign])

    # implementation for the sparse CSR representation completely written in
    # cython
    if sp.issparse(X):
        return inertia, _k_means._mini_batch_update_csr(
            X, x_squared_norms, centers, counts, nearest_center,
            old_center_buffer, compute_squared_diff)

    # dense variant in mostly numpy (not as memory efficient though)
    k = centers.shape[0]
    squared_diff = 0.0
    for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        count = center_mask.sum()

        if count > 0:
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            # inplace remove previous count scaling
            centers[center_idx] *= counts[center_idx]

            # inplace sum with new points members of this cluster
            centers[center_idx] += np.sum(X[center_mask], axis=0)

            # update the count statistics for this center
            counts[center_idx] += count

            # inplace rescale to compute mean of all points (old and new)
            centers[center_idx] /= counts[center_idx]

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return inertia, squared_diff

def _labels_inertia(X, x_squared_norms, centers,
                    precompute_distances=True, distances=None):
    """E step of the K-means EM algorithm.
    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.
    Parameters
    ----------
    X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.
    x_squared_norms: array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.
    centers: float64 array, shape (k, n_features)
        The cluster centers.
    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).
    distances: float64 array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.
    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment
    inertia : float
        Sum of distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    # distances will be changed in-place
    if sp.issparse(X):
        inertia = _k_means._assign_labels_csr(
            X, x_squared_norms, centers, labels, distances=distances)
    else:
        if precompute_distances:
            return _labels_inertia_precompute_dense(X, x_squared_norms,
                                                    centers, distances)
        inertia = _k_means._assign_labels_array(
            X, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia

def _init_centroids(X, k, init, random_state=None, x_squared_norms=None,
                    init_size=None):
    """Compute the initial centroids
    Parameters
    ----------
    X: array, shape (n_samples, n_features)
    k: int
        number of centroids
    init: {'k-means++', 'random' or ndarray or callable} optional
        Method for initialization
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    x_squared_norms:  array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None
    init_size : int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.
    Returns
    -------
    centers: array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.random_integers(
            0, n_samples - 1, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, string_types) and init == 'k-means++':
        centers = _k_init(X, k, random_state=random_state,
                          x_squared_norms=x_squared_norms)
    elif isinstance(init, string_types) and init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        centers = init
    elif callable(init):
        centers = init(X, k, random_state=random_state)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    _validate_center_shape(X, k, centers)
    return centers

def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++
    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters: integer
        The number of seeds to choose
    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state: numpy.RandomState
        The generator used to initialize the centers.
    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features))

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers