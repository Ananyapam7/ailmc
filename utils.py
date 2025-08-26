import jax
import jax.lax as lax
import jax.random as jr
from jax import jit
from jax.tree_util import tree_map, tree_structure, tree_unflatten
from jax.experimental import io_callback


def random_split_like_tree(rng_key, target):
    treedef = tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    out_key, rng_key = jr.split(rng_key, 2)
    keys_tree = random_split_like_tree(rng_key, target)
    return out_key, tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


from jax import lax, jit
from jax.experimental import io_callback

def _print_consumer(arg):
    iter_num, num_samples = arg
    print(f"Iter {iter_num:,} / {num_samples:,}")

@jit
def progbar(arg, result):
    iter_num, num_samples, print_rate = arg
    result = lax.cond(
        iter_num % print_rate == 0,
        lambda _: (io_callback(_print_consumer, None, (iter_num, num_samples)), result)[1],
        lambda _: result,
        operand=None,
    )
    return result


def progress_bar_scan(num_samples):
    """
    Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
    Note that `body_fun` must be looping over `jnp.arange(num_samples)`.
    This means that `iter_num` is the current iteration number
    """

    def _progress_bar_scan(func):
        print_rate = int(num_samples / 10)

        def wrapper_progress_bar(carry, iter_num):
            iter_num = progbar((iter_num + 1, num_samples, print_rate), iter_num)
            return func(carry, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan
