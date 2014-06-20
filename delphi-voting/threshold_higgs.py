#Uses a voting technique that I don't think has a name, so call it Delphi Method Voting
#Born from the need to apply thresholding: Change weakest backgrounds to signals, change weakest signals to backgrounds
#But this can't be done, because our best model (lm5.csv) did not have a correct RankOrder to apply threshold
#So now in pseudo-code:
#Best Model = lm5.csv
#Ensemble = [ lm5, lm6, lm7, xgboost3.67, nn_sub]
#If Best Model predicts Signal then predict Signal
#If Best Model predicts Background then:
#	Check if majority vote of Ensemble predicts Signal, if yes:
#		Change prediction to signal
#	Else:
#		Predict background
#Input: Location to best submission, glob (*.csv) of ensemble, Location to output submission
#Output: Number of backgrounds flipped to signals, and a .csv file

from collections import defaultdict
from glob import glob

def vote_background_to_signal(best_submission_loc, backup_submissions_glob, outfile_loc):
	signal_preds = defaultdict(int)
	with open(outfile_loc,"wb") as outfile:
		glob_files = glob(backup_submissions_glob)
		for glob_file in glob_files:
			for e, line in enumerate(open(glob_file)):
				if e > 0:
					row = line.strip().split(",")
					if row[2] == "s":
						signal_preds[row[0]] += 1
					else:
						signal_preds[row[0]] += -1
		d = 0
		for e, line in enumerate(open(best_submission_loc)):
			if e == 0:
				outfile.write( line )
			else:
				row = line.strip().split(",")
				if row[2] == "s":
					outfile.write(line)
				else:
					if signal_preds[row[0]] > 0:
						d+=1
						outfile.write(row[0]+","+row[1]+","+"s\n")
						#print "gotya", line
					else:
						outfile.write(line)
		return str(d)+ " background changed into signal. Saved as "+ outfile_loc
		
print vote_background_to_signal("lm5.csv", "lm*.csv","threshold-lm2.csv")