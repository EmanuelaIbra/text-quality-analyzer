from gramformer import Gramformer

gf = Gramformer(models=1, use_gpu=False)

text = "I is testng grammar tool using python."

corrected = gf.correct(text, max_candidates=1)

print(corrected)