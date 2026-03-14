# OCI Deployment Walkthrough

## Part 1 — Provision the VM

1. OCI Console -> Compute -> Instances -> **Create Instance**
2. Change shape to **VM.Standard.A1.Flex** (Ampere ARM, Always Free-eligible), set **2 OCPU / 12GB RAM** by expanding the row with the arrow next to the shape name
3. Change OS to **Oracle Linux 8**
4. Under networking, select **Create new virtual cloud network** and **Create new public subnet** — give them meaningful names e.g. `vcn-staves-app` and `subnet-public-staves-app`
5. Under SSH keys, select **Generate a key pair** and download the private key immediately — save to `~/.ssh/` and set permissions: `chmod 400 ~/.ssh/your-key.pem`
6. Click Create

> **Potential Issue — Out of Capacity:** Free tier accounts frequently hit "Out of host capacity" errors across all availability domains. The most reliable fix is upgrading to a **Pay As You Go (PAYG)** account. This requires a credit card verification (temporary $100 hold that is reversed), but does not remove Always Free benefits. A single 2 OCPU / 12GB A1.Flex instance costs $0/month as long as you stay within free tier limits. Set a billing alert at $1 to catch any accidental charges.

> **Potential Issue — Public IPv4 greyed out during creation:** During VM creation the "Automatically assign public IPv4" option may be greyed out even after selecting a public subnet. This is a UI bug — proceed with creation anyway. After the VM is created, assign the public IP manually: Instance details -> Attached VNICs -> click the VNIC -> IPv4 Addresses -> three dots next to the private IP -> Edit -> assign Ephemeral public IP.

---

## Part 2 — Configure Networking

1. After assigning the ephemeral public IP, go to Instance details -> **Quick Actions** -> **Connect public subnet to internet** -> click Connect
   - This sets up the internet gateway and route table rules
   - You may see a warning that the VCN already has an internet gateway — this is fine, it will reuse it and just configure the route rules

2. Open port 8000 in the **OCI Security List**:
   - Networking -> Virtual Cloud Networks -> `vcn-staves-app` -> Subnets -> `subnet-public-staves-app` -> click the Security List
   - Add Ingress Rule: Source CIDR `0.0.0.0/0`, Protocol TCP, Destination Port `8000`

> **Known Issue — Site can't be reached after deploying container:** If your container is running but the public URL is unreachable, the issue is almost always the OCI Security List. Verify by SSH-ing into the VM and running `curl http://localhost:8000/docs` — if that returns HTML then the container is fine and only the OCI Security List rule is missing. The Network Security Group (NSG) and Security List are two separate things in OCI — both can block traffic independently. Make sure the ingress rule is added to the **Security List linked to the subnet**, not just the NSG.

---

## Part 3 — Install Docker on the VM

SSH into the VM:

```bash
ssh -i ~/.ssh/your-key.pem opc@<public-ip>
```

> **Note:** Default user on Oracle Linux is `opc`, not `ubuntu`

Install Docker CE (Oracle Linux ships with `podman-docker` which must be removed first):

```bash
sudo dnf remove -y podman-docker
sudo dnf install -y dnf-utils
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world  # verify
```

Open port 8000 in the VM firewall:

```bash
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

---

## Part 4 — Set Up OCI Container Registry (OCIR)

1. OCI Console -> Developer Services -> **Container Registry** -> Create Repository
   - Name: `staves-detector`, Access: **Private**
2. Find your **tenancy namespace**: Profile -> Tenancy -> Object Storage Namespace
3. Your registry URL: `<region-key>.ocir.io/<tenancy-namespace>/staves-detector`
   - e.g. `phx.ocir.io/axgy0b9smctc/staves-detector` for US West Phoenix
4. Generate an **Auth Token**: Profile -> My Profile -> Auth Tokens -> Generate Token — copy immediately, it will not be shown again

Region key reference:

| Region | Key |
|---|---|
| US East (Ashburn) | `iad` |
| US West (Phoenix) | `phx` |
| UK South (London) | `lhr` |
| Germany Central (Frankfurt) | `fra` |
| Japan East (Tokyo) | `nrt` |

---

## Part 5 — Build and Push Docker Image

On your **local machine**, log into OCIR:

```bash
docker login phx.ocir.io \
  -u <tenancy-namespace>/<oci-username> \
  -p 'your-auth-token'
```

> **Note:** Wrap the auth token in single quotes — special characters like `)`, `>`, `+` will cause bash syntax errors if unquoted

Build for ARM64 (required for A1.Flex VM):

```bash
docker build \
  --platform linux/arm64 \
  -t phx.ocir.io/<tenancy-namespace>/staves-detector:latest .
```

> **Note:** Building for ARM64 on an x86 machine uses QEMU emulation and is significantly slower than a native build. A step with large apt or pip installs (e.g. ffmpeg) can take 3-5 minutes. This is normal — do not interrupt it.

Push to OCIR:

```bash
docker push phx.ocir.io/<tenancy-namespace>/staves-detector:latest
```

---

## Part 6 — Deploy on the VM

SSH into the VM and log into OCIR:

```bash
docker login phx.ocir.io \
  -u <tenancy-namespace>/<oci-username> \
  -p 'your-auth-token'
```

Pull and run the container:

```bash
docker pull phx.ocir.io/<tenancy-namespace>/staves-detector:latest

docker run -d \
  --name staves-detector \
  --restart unless-stopped \
  -p 8000:8000 \
  phx.ocir.io/<tenancy-namespace>/staves-detector:latest
```

Verify it is running:

```bash
docker ps
docker logs staves-detector
curl http://localhost:8000/docs  # should return HTML
```

Your API is live at:

```
http://<vm-public-ip>:8000
http://<vm-public-ip>:8000/docs  <- FastAPI Swagger UI
```

---

## Debugging Checklist

| Symptom | Likely Cause | Fix |
|---|---|---|
| Out of capacity on all ADs | Free tier capacity exhausted | Upgrade to PAYG |
| Public IPv4 greyed out during creation | OCI UI bug | Assign ephemeral IP post-creation via VNIC settings |
| Site can't be reached | OCI Security List missing port rule | Add TCP 8000 ingress rule to subnet's Security List |
| `docker run hello-world` pulls every time | No local cache | Normal on first run |
| Auth token causes bash syntax error | Special characters in token | Wrap token in single quotes |
| ARM build is very slow | QEMU emulation on x86 host | Normal — leave it running |

---

> **Security reminder:** Never commit your OCI private key, auth token, or any credentials to your GitHub repo. Store private keys in `~/.ssh/` with `chmod 400` permissions. Store all tokens in a `.env` file that is gitignored.