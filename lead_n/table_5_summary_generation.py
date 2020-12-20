# Generation of summary given in table 5 

import rouge 

rouge = rouge.Rouge()
ref = "i have one year of responsible credit history and i want a chase sapphire card. is that enough to qualify me?"
gen = "hi reddit! i wanted some insight on the requirements to get a chase sapphire credit card."
scores = rouge.get_scores(gen, ref, avg=False)

print(scores)
