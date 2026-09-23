FROM python:3.11-slim

# Non-root user is standard Docker practice, and gives sentence-transformers
# a real $HOME to cache downloaded models under (~/.cache/huggingface).
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
# Install the CPU-only torch build FIRST, from its own dedicated index,
# before anything that depends on it. The default PyPI torch wheel for
# Linux declares hard dependencies on several nvidia-*-cu12 CUDA packages
# (several hundred MB) that are dead weight on Render's CPU-only free tier
# -- this app never touches a GPU -- and that unused bulk was pushing
# resident memory over the 512MB limit during model loading. Doing this as
# a separate, prior step (rather than --extra-index-url alongside PyPI on
# the requirements.txt install) is deliberate: pip's resolver doesn't
# guarantee which index wins when a package exists on both, so installing
# torch on its own first, from --index-url (not --extra-index-url), is the
# only way to be certain the CPU build is what's actually present when
# sentence-transformers is installed next and finds its torch requirement
# already satisfied.
RUN pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Bake both models into the image at build time instead of letting
# sentence-transformers download them on first use. Free-tier instances
# spin down after inactivity and have no persistent disk, so without this
# every cold start would re-download ~200MB+ of models before it could
# serve a single request -- baking them in trades a slower one-time build
# for a fast wake-up on every subsequent cold start.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user app.py app.py
COPY --chown=user products.json products.json
COPY --chown=user templates/ templates/

# Render assigns $PORT dynamically at container start and routes traffic to
# whatever it's set to; 7860 is just the local-run fallback if $PORT isn't set.
ENV PORT=7860
EXPOSE 7860

# gunicorn, not `python app.py` -- Flask's built-in server (and its debug
# mode) isn't meant for anything public-facing. One worker: each worker is a
# separate process with its own copy of the loaded models (~roughly 200MB+),
# and the free tier's RAM is limited. Generous timeout to leave room for a
# cold container's first request landing while imports/model loads are still
# settling. Shell form (not exec-array form) so $PORT actually expands --
# an exec-form CMD passes "${PORT:-7860}" to gunicorn literally, unexpanded.
CMD sh -c 'gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 120 app:app'
