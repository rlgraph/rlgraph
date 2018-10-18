### RLGraph - Contrib Policies

Any code in this directory is not officially supported, and may change
or be removed at any time without notice.

The `contrib` directory contains project directories, each of which has
designated owners (e.g. you). It is meant to contain new features or
great-idea prototype contributions for RLGraph, all of which may
eventually get merged into RLGraph itself.
The interfaces of features in `contrib` may change and also require
further testing to see whether they can find broader user acceptance.
As we are trying to keep duplication within `contrib` to a minimum,
you may be asked to 1) refactor code in `contrib` and 2) use some
feature inside core RLGraph or in another project in contrib (rather than
reimplementing/reinventing the feature).

When adding a project, please abide to the following simple directory
structure rules:

1) Create a project directory inside `contrib/` (chose any name that
does not exist yet, e.g. `my_project`).

2) Mirror the portions of the RLGraph tree that your project requires
underneath `contrib/my_project/`.
For example, let's say you create a new "Filter" Component in the file:
`my_filter.py`. If you were to merge this file directly
into RLGraph, it would live in
`rlgraph/components/filters/my_filter.py`.
In `contrib/`, it is part of project `my_project`, and therefore, its
full path must be:
`contrib/my_project/rlgraph/components/filters/my_filter.py`.


Have fun RLGraph-ing :)

And thank you for your contributions to this library!

