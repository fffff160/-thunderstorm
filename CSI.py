#!/usr/bin/python
import numpy as np
import logging

def find_best_CSI(dec_valuee, labels, confident=0):
	'''go through all the confident level, find the best CSI when POD >= 0.6
		return best CSI'''

	# the first dimension of dec_value must match the labels' dimension
	if dec_valuee.shape[0] != labels.shape[0]:
		print 'dimensions don not match'
		return

	# get first column of dec_value
	dec_value = dec_valuee[:, 0][:]
	num = dec_value.shape[0]
	
	# init
	best_CSI = 0
	TP = FP = FN = TN = 0
	###############################################################
	if confident == 0:
		# 1.
		# confident level go through [0.1,0.9]. The stride is 0.1
		for temp_conf in np.linspace(0.1, 0.9, 9):

			# use confident level to compute dec_labels
			dec_labels = np.ones( num, dtype=np.int )
			dec_labels[ ( dec_value >= temp_conf ) ] = 0

			# compute the CSI
			CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc = get_CSI( dec_labels, labels )

			# find best CSI
			if ( POD > 0.6 ) and ( CSI > best_CSI ):
				best_CSI = CSI
				confident = temp_conf

	###############################################################

		# 2.
		# confident level go through boundary. The stride is 0.001

		# compute the boundary
		left_margin = confident - 0.2
		if left_margin <= 0:
			left_margin =  0

		right_margin = confident + 0.2
		if right_margin >= 1:
			left_margin =  1

		temp_conf = left_margin
		while temp_conf <= right_margin:

			# use confident level to compute dec_labels
			dec_labels = np.ones( num, dtype=np.int )
			dec_labels[ ( dec_value >= temp_conf ) ] = 0

			# compute the CSI
			CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc= get_CSI( dec_labels, labels )

			# find best CSI
			if ( POD > 0.6 ) and ( CSI > best_CSI ):
				best_CSI = CSI
				confident = temp_conf

			temp_conf += 0.001

###############################################################

	# 3.
	# use the proper confident to get final output

	# use confident level to compute dec_labels
	dec_labels = np.ones( num, dtype=np.int )
	dec_labels[ ( dec_value >= confident ) ] = 0

	# compute the CSI
	CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc= get_CSI( dec_labels, labels )
	

	# print the output
	print 'CSI %5.3f, POD %5.3f, FAR  %5.3f, Confident %5.3f' \
		%(CSI, POD, FAR, confident)
	print ' TP, FP, FN, TN ',( TP, FP, FN, TN )
	print ' accuracy: %5.3f ' %(acc)
	print ' CI_accuracy: %5.3f ' %(CI_acc)
	print ' NCI_accuracy: %5.3f ' %(NCI_acc)

	logging.info('CSI %5.3f, POD %5.3f, FAR  %5.3f, Confident %5.3f'
		% (CSI, POD, FAR, confident))
	logging.info(' TP, FP, FN, TN: %d, %d, %d, %d' % (TP, FP, FN, TN))
	logging.info(' accuracy: %5.3f ' % acc)
	logging.info(' CI_accuracy: %5.3f ' % CI_acc)
	logging.info(' NCI_accuracy: %5.3f ' % NCI_acc)

	return [CSI, POD, FAR, confident]

	
#####################################################################################################

def get_CSI(dec_labels, true_labels):
	'''calculate the CSI, POD, FAR and return them'''

	# number of samples
	num = dec_labels.shape[0]
	
	# compute TP, FP, FN, TN
	TP = float( np.sum( true_labels[dec_labels == true_labels] ) )
	FP = float( np.sum( dec_labels == 1 ) - TP )
	FN = float( np.sum( true_labels == 1 ) - TP )
	TN = float( np.sum( true_labels[dec_labels == true_labels] == 0 ) )

	# compute CSI, POD, FAR	
	if TP + FP + FN == 0:
		print 'There is no CI'
		CSI = 0
	else:
		CSI = TP / ( TP + FP + FN ) 

	if TP + FN == 0:
		print 'There is no CI'
		POD = 0
	else:
		POD = TP / ( TP + FN ) 

	if FP + TP == 0:
		FAR = 0
	else:
		FAR = FP / ( FP + TP )

	# compute CI, NCI accuracy	
	acc = (TP + TN) / (TP + TN + FP + FN)
	CI_acc = TP / ( TP + FN )
	NCI_acc = TN / ( TN + FP )	
	
	'''
	# print	
	print 'TP', TP
	print 'FP', FP
	print 'FN', FN
	print 'TN', TN	
	print '%5.3f %5.3f %5.3f' %( CSI, POD, FAR )	
	'''

	return [CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc]

#####################################################################################################

