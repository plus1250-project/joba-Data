## Settings
### Airflow

**1. Install Pre-requisites**
```
sudo yum install gcc gcc-c++ -y
sudo yum install libffi-devel mariadb-devel cyrus-sasl-devel -y
sudo dnf install redhat-rpm-config
```
  
**2. Install Anaconda3**
```
sudo yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh 
./Anaconda3-2020.02-Linux-x86_64.sh
```
  
**3. Install Apache Airflow**
```
sudo pip3 install apache-airflow[mysql,celery]
```
  
**4. Initialize Airflow**
```
export AIRFLOW_HOME=~/airflow
airflow initdb
```
  
**5. Install MySQL Server**
```
sudo rpm -Uvh https://repo.mysql.com/mysql80-community-release-el7-3.noarch.rpm
sudo sed -i 's/enabled=1/enabled=0/' /etc/yum.repos.d/mysql-community.repo
sudo yum --enablerepo=mysql80-community install mysql-server
sudo systemctl start mysqld.service
```
  
**6. Check temporary password and Login to MySQL**
```
sudo grep 'temporary password' /var/log/mysqld.log
mysql -u root -p
```
  
**7. Configure database for Airflow**
```
create database airflow;
create user userID@'client' identified by 'PW';
grant all privileges on airflow.* to userID@'client' identified by 'PW';
flush privileges;
```
  
**8. Update Airflow configuration file (~/airflow/airflow.cfg)**
```
sql_alchemy_conn = mysql://userID:PW@localhost:3306/airflow
executor = CeleryExecutor
```
  
**9. Initialize Airflow**
```
airflow initdb
```
  
**10. Create Airflow account**
```
airflow users create\
--username [user name]\
--firstname [user firstname]\
--lastname [user lastname]\
--role [user role]\
--email [user email]
```
