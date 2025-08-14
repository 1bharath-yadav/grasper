FROM python:3.12-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
    
RUN useradd -m -u 1000 user

COPY --chown=user . /home/user/app

# Environment setup
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/home/user/.local/bin:$PATH"


# Install Python dependencies via uv
RUN uv sync --no-cache-dir

# Make run script executable
RUN chmod +x run.sh

# Expose app ports
EXPOSE 7860
EXPOSE 8501

USER user

ENTRYPOINT ["bash", "./run.sh"]
