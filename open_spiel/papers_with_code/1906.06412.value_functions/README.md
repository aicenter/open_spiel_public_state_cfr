# 2021-09-10

- zasumene values nepomahaju
- all particles ma podobnu perf ako 64 particles
- "lost in noise"
- oracle bot values training - wip,
- RBC bug fixes
- mean pooling ~2x vacsia loss ale podobna perf.

- regresne testy -- trunk expl
- pozicne neuronky
- validacna loss - nahodna, oracle
- CFR-AVE s neuronkami

- existuje hra kde priemer values nie je to spravne
- resolving vzdy so spravnymi values
- trunk oracle


# 2021-09-07

- oracle: nemam jednoznacny zaver na particle selection podla max q-values
  a public state minimax value approx - IIGS(3) resp. IIGS(4) dava vahu naopak.
- zasumene values nepomahaju
- all particles ma podobnu perf ako 64 particles
- "lost in noise" 
- oracle bot values training - wip, 
- RBC bug fixes
- mean pooling ~2x vacsia loss ale podobna perf.

- regresne testy -- trunk expl
- pozicne neuronky
- validacna loss - nahodna, oracle
- CFR-AVE s neuronkami

- existuje hra kde priemer values nie je to spravne
- resolving vzdy so spravnymi values
- trunk oracle

---

- expected values against uniform
- tabularize IS-MCTS strategies with some computational time, calculate expected values
- tabularization: additional mass not uniform, but lowest card?
- oracle (5k subgame iters) with particles has high exploitability :-/


- po prvom kroku kolko
---

Are we leaking memory?

Make reasonable perf on IIGS(3,6)
- Make ISMCTS/random metric
- Improve BR time
---

Debts
- [x] Setup proper tests for previous results -- had some regression bugs.
- [x] Oracle values and use all particles vs subset of them and with(out) bottleneck input sizes for regression.

Technicalities
- [x] Setting up buffer sizes so we can always do VF evaluation (limited branching factor).
- [ ] Parallelization: data generation and evaluation against IS-MCTS.
- [ ] Improve logging for metrics (so we can have outputs before the experiment is over).
- [ ] More efficient tabularization of the online algorithm.               (expl on IIGS-4 should be enough)

Making sure things work well
- [x] Continual resolving - REBEL-style with full VFs (works with cfr on Kuhn, use the net as well) -- David?
- [x] Average values for CR from more recent iterations.
- [ ] More than 1-step lookahead for online play?

Practical perf on IIGS
- [x] IS-MCTS bootstrapping.
- [x] IS-MCTS (and/or random) evaluation.

Idea verification
- [?] Find out the size of cf. supports in IIGS.
We can have UB with some Nash
Advancements
- [ ] Assign beliefs for regenerated particles.
Recomputing in online play -- Uniform (because the opponent made a mistake to get there and hence there is a meaningful refinement with uniform)

Networks
- [ ] Value net uses information about the set of particles.
- [ ] Transformers for change-of-basis / regression.
- [x] Average/max pooling, some other methods of pooling?

Evaluation Domains
- [x] Ca we make IIGS-(100,4) (4 rounds with 100 cards)
- [ ]  Domain specific exploitability, opponent 2K highest cards, player k+1 (“pass” card) cards with small probs collapsed to the smallest card

Brian’s paper
[X] 1-step CK size in IIGS? Implementation?
Likely yes

[x] Writing
[x] pseudocode
Game where we can’t do it with CFR / Rebel from some K cards
Argument preco to vieme aj v dalsich hrach -- abstrakcia, bucketing, vziat najvyssi bucket

----


Towards DarkChess:

- [ ] Update training set to use sequences
- [ ] New model that uses sequences
- [ ] Train and run.

Možné zlepšenia:

- [ ] PCFR+
- [ ] Smoothed-out CFR-D ?
- [ ] Resolving games ? Rebel-style randomizations?
- [ ] Referee for bot observations
- [ ] Parallel evaluation of public states within a batch
- [ ] Transformers for change-of-basis / regression
- [ ] Two-pooling of beliefs for each player and then concat
 
--

Tyzden, na Arxiv

- [ ] Porazit random
- [ ] Staticky training set
- [ ] Public features in the particle model.
- [ ] Tabularizacia online algoritmu


- [ ] POMCP na battleship  -- conf/journal?

- Rebel-style randomizations? -- mohlo by to pomoct, lebo nevieme cfv constrains
- Generalizacia values -- najst najhorsie kompatib. cfvs nad novymi rnagmie


machine   cpu   gpu
------------------------------------
ursa     504    0
glados    40    1x 1080Ti GPU
black     24    4x Tesla P100
zia      128    4x A100
adan      32    2x Tesla T4 16GB
doom      16    2x Tesla K20 5GB
grimbold  32    2x Tesla P100 12GB
cha       32    8x GeForce RTX 2080 Ti
fau       64    ?x Quadro RTX 5000
konos     20    4x GeForce GTX 1080 Ti




--------------------------------------------------------------------------------

- [x] utilities reloading in infostate tree
- [x] public state features "reloading"
- [ ] TerminalPublicStateContext utilities reloading
