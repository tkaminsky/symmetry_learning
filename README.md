# symmetry_learning

This library implements a basic Gym environment whose goal is to match nested symmetric shapes


## Getting started

<code>symmetry_shift_runner.py</code> is the go-to file for running experiments. Right now there are options for manual control, an oracle, or for a BC model to train.

Here's an example code for running with a bc agent:

<code>
python symmetry_shift_runner.py -c bc
</code>

Feel free to change any experiment parameters in algos/bc.


## Blueprints

A nested polygon blueprint tells how to build a shape of nested symmetries.

Depending on whether it's regular or generic, you may need to specify a different set of data in each blueprint.

See blueprints.py for some examples of each type.

## Environment

<code>environment/symmetry_shift.py</code> contains gym environment for trying to line up one subshape from a nested object, with error given by the nested polygon distance between the objects (feel free to change this to use another metric---this implementation is a little fiddly).

Right now, you have to specify the two objects in the init, but it should be easy enough to add a function to change the active blueprint.

## Contact

Feel free to slack me or shoot me an email at tomdavkam@gmail.com if you have any questions.