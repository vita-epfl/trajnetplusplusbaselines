.. image:: https://travis-ci.org/svenkreiss/socialforce.svg?branch=master
    :target: https://travis-ci.org/svenkreiss/socialforce


Social Force Model
==================

.. code-block::

    Social force model for pedestrian dynamics
    Dirk Helbing and Péter Molnár
    Phys. Rev. E 51, 4282 – Published 1 May 1995


Install and Run
===============

.. code-block:: sh

    # install from PyPI
    pip install 'socialforce[test,plot]'

    # or install from source
    pip install -e '.[test,plot]'

    # run linting and tests
    pylint socialforce
    pytest tests/*.py


Ped-Ped-Space Scenarios
=======================

+----------------------------------------+----------------------------------------+
| .. image:: docs/separator.gif          | .. image:: docs/gate.gif               |
+----------------------------------------+----------------------------------------+
| Emergent lane formation with           | Emergent lane formation with           |
| 30 pedestrians:                        | 60 pedestrians:                        |
|                                        |                                        |
| .. image:: docs/walkway_30.gif         | .. image:: docs/walkway_60.gif         |
+----------------------------------------+----------------------------------------+


Ped-Ped Scenarios
=================

+----------------------------------------+----------------------------------------+
| .. image:: docs/crossing.png           | .. image:: docs/narrow_crossing.png    |
+----------------------------------------+----------------------------------------+
| .. image:: docs/opposing.png           | .. image:: docs/2opposing.png          |
+----------------------------------------+----------------------------------------+
