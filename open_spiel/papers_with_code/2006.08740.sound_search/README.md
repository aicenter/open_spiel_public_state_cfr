# Sound Search in Imperfect Information Games

Arxiv link: https://arxiv.org/pdf/2006.08740.pdf

## Abstract

Search has played a fundamental role in computer game research since the very beginning. And while online search has been commonly used in perfect information games such as Chess and Go, online search methods for imperfect information games have only been introduced relatively recently.This paper addresses the question of what is sound search in an imperfect information setting of two-player zero-sum games?  We argue that the fixed-strategy definitions of exploitability and epsilon-Nash equilibria are ill suited to measure the worst-case performance of an online search algorithm.We thus formalize epsilon-soundness, a concept that connects the worst-case performance of an online algorithm to the performance of an epsilon-Nash equilibrium.  As epsilon-soundness can be difficult to compute in general, we also introduce a consistency framework – a hierarchy that connects the behavior of an online algorithm to a Nash equilibrium. Our multiple levels of consistency describe in what sense an online algorithm plays “just like a fixed Nash equilibrium”. These notions further illustrate the difference in perfect and imperfect information settings, as the same consistency guarantees have different worst-case online performance in perfect and imperfect information games. Our definition of soundness and the consistency hierarchy finally provide appropriate tools to analyze online algorithms in imperfect information games. We thus inspect some of the previous online algorithms in a new light, bringing new insights into their worst case performance guarantees.

## Steps to reproduce experiments

Follow the standard OpenSpiel build instructions. 

Install [GNU parallel](http://www.gnu.org/software/parallel/).

Then:

```
    $ cd build/
    $ make tabularize_oos  # Compile the tabularize_oos target.
    $ cd ../open_spiel/papers_with_code/2006.08740.sound_search
    $ ./generate_data.sh
    $ python plot_coordinated_mp.py
    $ python plot_coordinated_mp_appendix.py
```

Example output for 1 seed is already saved in the *.csv files.
