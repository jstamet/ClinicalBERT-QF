# ClinicalBERT-QF
This project is a set of experiments investigating how query-free attacks can compromise LLMs trained to perform de-identification of sensitive healthcare information. Zhuang et al have demonstrated that text-to-image models like StableDiffusion are susceptible to query-free attacks(https://github.com/OPTML-Group/QF-Attack/). This project applies similar techniques to simulate attacks against masked language models like ClinicalBERT.


Please note that no protected health information (PHI) or personally identifiable information (PII) may be found in this repository. All examples of PHI are realistic but fake examples generated from some model like Claude or GPT. 

<div align="center">
  <img src="/images/homoglyph.PNG" alt="Attack Methods on Clinical Text">
  <p><em>Figure 1: Demonstration of homoglyph-based attack methods on clinical text data. Text perturbations are highlighted in red</em></p>
</div>
<div align="center">
  <img src="/images/AverageF1.png" alt="Attack Methods on Results">
  <p><em>Figure 2: Average F1 scores showing how effectively attacks impact classification of PHI (lower score means more effective attack)</em></p>
</div>

