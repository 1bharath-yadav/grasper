FROM python:3.12-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set working directory for the app
WORKDIR /home/user/app

# Copy application files and set ownership to the non-root user
COPY --chown=user . /home/user/app

# Download uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Install uv into the user's local bin
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Environment setup - make sure uv is in the PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install Python dependencies with uv (creates .venv owned by user)
RUN uv sync --no-cache-dir

# Make the run script executable
RUN chmod +x run.sh

# Expose application ports
EXPOSE 7860
EXPOSE 8501

# Start the app
ENTRYPOINT ["bash", "./run.sh"]

