__global__ void compute_mass_matrix(double *values, int* rows, int* cols, int res, double d, double* vertices, int *tets, int tets_num)
{
    const int idx = threadIdx.x + blockIdx.y*res + blockIdx.x*res*res;
    if (idx >= tets_num)
        return;
    tets = tets + idx*4;
    double x[4],y[4],z[4];
    for (int i = 0;i < 4;i++){
        x[i] = vertices[tets[i]*3];
        y[i] = vertices[tets[i]*3 + 1];
        z[i] = vertices[tets[i]*3 + 2];
    }
    int vid[12] = {tets[0]*3,tets[0]*3+1,tets[0]*3+2,
                    tets[1]*3,tets[1]*3+1,tets[1]*3+2,
                    tets[2]*3,tets[2]*3+1,tets[2]*3+2,
                    tets[3]*3,tets[3]*3+1,tets[3]*3+2,
                    };
    double V =((x[1] - x[0])*((y[2] - y[0])*(z[3] - z[0])-(y[3] - y[0])*(z[2] - z[0]))+(y[1] - y[0])*((x[3] - x[0])*(z[2] - z[0])-(x[2] - x[0])*(z[3] - z[0]))+(z[1] - z[0])*((x[2] - x[0])*(y[3] - y[0])-(x[3] - x[0])*(y[2] - y[0])))/6;
    V = abs(V);
    double m[] = {2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0
                ,0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0
                ,0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1
                ,1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0
                ,0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0
                ,0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1
                ,1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0
                ,0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0
                ,0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1
                ,1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0
                ,0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0
                ,0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2};

    
    for (int i = 0;i < 12;i++)
        for(int j = 0;j < 12;j++)
            m[i*12+j] *= (d/20)*V;

    int  offset = idx*12*12;
    for (int i = 0;i < 12; i++)
        for(int j = 0;j < 12; j++){
            values[offset + i*12 + j] = m[i*12 + j];
            rows[offset + i*12 + j] = vid[i];
            cols[offset + i*12 + j] = vid[j];
        }
}


__global__ void compute_stiff_matrix(double *values, int* rows, int* cols, int res, double E0, double v, double* vertices, int *tets, int tets_num)
{
    const int idx = threadIdx.x + blockIdx.y*res + blockIdx.x*res*res;
    if (idx >= tets_num)
        return;
    tets = tets + idx*4;
    double x[4],y[4],z[4];
    for (int i = 0;i < 4;i++){
        x[i] = vertices[tets[i]*3];
        y[i] = vertices[tets[i]*3 + 1];
        z[i] = vertices[tets[i]*3 + 2];
    }
    int vid[12] = {tets[0]*3,tets[0]*3+1,tets[0]*3+2,
                    tets[1]*3,tets[1]*3+1,tets[1]*3+2,
                    tets[2]*3,tets[2]*3+1,tets[2]*3+2,
                    tets[3]*3,tets[3]*3+1,tets[3]*3+2,
                    };
    double a[4],b[4],c[4],V;
    a[0]=y[1]*(z[3] - z[2])-y[2]*(z[3] - z[1])+y[3]*(z[2] - z[1]);
    a[1]=-y[0]*(z[3] - z[2])+y[2]*(z[3] - z[0])-y[3]*(z[2] - z[0]);
    a[2]=y[0]*(z[3] - z[1])-y[1]*(z[3] - z[0])+y[3]*(z[1] - z[0]);
    a[3]=-y[0]*(z[2] - z[1])+y[1]*(z[2] - z[0])-y[2]*(z[1] - z[0]);
    b[0]=-x[1]*(z[3] - z[2])+x[2]*(z[3] - z[1])-x[3]*(z[2] - z[1]);
    b[1]=x[0]*(z[3] - z[2])-x[2]*(z[3] - z[0])+x[3]*(z[2] - z[0]);
    b[2]=-x[0]*(z[3] - z[1])+x[1]*(z[3] - z[0])-x[3]*(z[1] - z[0]);
    b[3]=x[0]*(z[2] - z[1])-x[1]*(z[2] - z[0])+x[2]*(z[1] - z[0]);
    c[0]=x[1]*(y[3] - y[2])-x[2]*(y[3] - y[1])+x[3]*(y[2] - y[1]);
    c[1]=-x[0]*(y[3] - y[2])+x[2]*(y[3] - y[0])-x[3]*(y[2] - y[0]);
    c[2]=x[0]*(y[3] - y[1])-x[1]*(y[3] - y[0])+x[3]*(y[1] - y[0]);
    c[3]=-x[0]*(y[2] - y[1])+x[1]*(y[2] - y[0])-x[2]*(y[1] - y[0]);
    V =((x[1] - x[0])*((y[2] - y[0])*(z[3] - z[0])-(y[3] - y[0])*(z[2] - z[0]))+(y[1] - y[0])*((x[3] - x[0])*(z[2] - z[0])-(x[2] - x[0])*(z[3] - z[0]))+(z[1] - z[0])*((x[2] - x[0])*(y[3] - y[0])-(x[3] - x[0])*(y[2] - y[0])))/6;
    V = abs(V);
    double e[] = {1-v ,v   ,v   ,0    ,0    ,0    
                ,v   ,1-v ,v   ,0    ,0    ,0    
                ,v   ,v   ,1-v ,0    ,0    ,0    
                ,0   ,0   ,0   ,0.5-v,0    ,0    
                ,0   ,0   ,0   ,0    ,0.5-v,0    
                ,0   ,0   ,0   ,0    ,0    ,0.5-v};
    for(int i = 0;i < 6*6;i++)
        e[i] = e[i] *(E0/(1+v)/(1-2*v));
    

    double be[] = {a[0],0   ,0   ,a[1],0   ,0   ,a[2],0   ,0   ,a[3],0   ,0   
                 ,0   ,b[0],0   ,0   ,b[1],0   ,0   ,b[2],0   ,0   ,b[3],0   
                 ,0   ,0   ,c[0],0   ,0   ,c[1],0   ,0   ,c[2],0   ,0   ,c[3]
                 ,b[0],a[0],0   ,b[1],a[1],0   ,b[2],a[2],0   ,b[3],a[3],0   
                 ,0   ,c[0],b[0],0   ,c[1],b[1],0   ,c[2],b[2],0   ,c[3],b[3]
                 ,c[0],0   ,a[0],c[1],0   ,a[1],c[2],0   ,a[2],c[3],0   ,a[3]};
    for(int i = 0;i < 6*12;i++)
        be[i] = be[i] /(6*V);

    double ke1[12*6];
    for (int i = 0;i < 12;i++)
        for(int j = 0;j < 6;j++){
            ke1[i*6+j] = 0;
            for(int k = 0;k < 6;k++)
                ke1[i*6+j] += be[k*12+i]*e[k*6+j];
        }
    double ke2[12*12];

    for (int i = 0;i < 12;i++)
        for(int j = 0;j < 12;j++){
            ke2[i*12+j] = 0;
            for(int k = 0;k < 6;k++)
                ke2[i*12+j] += ke1[i*6+k]*be[k*12+j];
            ke2[i*12+j] *= V;
        }

    
    // for (int i = 0;i < 12;i++)
    //     for(int j = 0;j < 6;j++)
    //         ke2[i*12+j] = ke1[i*6+j];
    // for (int i = 0;i < 6;i++)
    //     for(int j = 0;j < 12;j++)
    //         ke2[i*12+j] = be[i*12+j];

    int  offset = idx*12*12;
    for (int i = 0;i < 12; i++)
        for(int j = 0;j < 12; j++){
            values[offset + i*12 + j] = ke2[i*12 + j];
            rows[offset + i*12 + j] = vid[i];
            cols[offset + i*12 + j] = vid[j];
        }

}