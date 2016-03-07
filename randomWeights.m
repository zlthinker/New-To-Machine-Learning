function W = randomWeights(rows, cols)

	thres = 0.01;
	W = 2 * rand(rows, cols + 1) * thres - thres;

end
