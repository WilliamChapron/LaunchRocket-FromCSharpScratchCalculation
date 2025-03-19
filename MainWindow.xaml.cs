using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using System.Windows.Threading;
using TpMath3d.InvertMatrix;

using System.Diagnostics;
using System.IO;

namespace TpMath3d
{
    public partial class MainWindow : Window
    {
        //All
        private double[,] points;
        private double[] pos;
        private double[] speed;
        private double hTimeStep;
        private MatrixOperations matrixOperations;
        private DispatcherTimer timer;

        private double mMass;

        // Translation
        private double[] sumForceTranslation;
        // Rotation
        private double[] rotAngle;
        private double[] rotSpeed;
        private double[,] forcesRotation; // Will be calculate by rotation fonction
        private double[,] applicationPointsRotation;

        private double[] inertieCenter;
        private double[,] inertieMatrix;

        private Process consoleProcess;
        private StreamWriter consoleWriter;

        public MainWindow()
        {
            matrixOperations = new MatrixOperations();
            // Cylinder
            double R = 3.0;
            double h = 5.0;
            double x0 = 0.0, y0 = 0.0, z0 = 0.0;
            int n = 10;
            int m = 20;
            points = matrixOperations.cylindre_plein(R, h, x0, y0, z0, n, m);
            int rowCount = points.GetLength(0);
            double[,] newPoints = new double[rowCount, 4];
            double masse = 1.0;

            for (int i = 0; i < rowCount; i++)
            {
                newPoints[i, 0] = points[i, 0];
                newPoints[i, 1] = points[i, 1];
                newPoints[i, 2] = points[i, 2];
                newPoints[i, 3] = masse;
            }
            points = newPoints;
            // ***********


            hTimeStep = 0.1; 
            mMass = 1.0;
            pos = new double[] { 0.0, 0.0, 0.0 }; 
            speed = new double[] { 0.0, 0.0, 0.0 }; 
            rotAngle = new double[] { 0.0, 0.0, 0.0 }; 
            rotSpeed = new double[] { 0.0, 0.0, 0.0 };

            //inertieCenter = matrixOperations.centre_inert(points);
            inertieCenter = new double[] { 0.0, 0, 0.0 };


            applicationPointsRotation = new double[,] { { -R,0, 0.0 }, { R, 0, 0.0 } };
            forcesRotation = new double[,] { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 3 } };
            sumForceTranslation = new double[] { 0.0, 0.0, 13.0/1000 };

            InitializeComponent();

            inertieMatrix = matrixOperations.matrice_inert(points);

            DrawPoints(points);


            timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromSeconds(hTimeStep); 
            timer.Tick += new EventHandler(Timer_Tick); 

            timer.Start();




        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            matrixOperations.translationPoints(mMass, hTimeStep, sumForceTranslation, ref points, ref speed);
            matrixOperations.rotationPoints(hTimeStep, forcesRotation, applicationPointsRotation, inertieCenter, inertieMatrix, ref rotAngle, ref rotSpeed, ref points);
            myViewport.Children.Clear();

            DrawPoints(points);
        }


        private void DrawPoints(double[,] points)
        {
            Model3DGroup group = new Model3DGroup();

            for (int i = 0; i < points.GetLength(0); i++)
            {
                double x = points[i, 0];
                double y = points[i, 1];
                double z = points[i, 2];

                group.Children.Add(CreatePointSphere(new Point3D(x, z-10, y), 0.1, Colors.Red));
            }

            ModelVisual3D model = new ModelVisual3D { Content = group };
            myViewport.Children.Add(model);
        }

        private GeometryModel3D CreatePointSphere(Point3D center, double size, Color color)
        {
            MeshGeometry3D mesh = new MeshGeometry3D();
            int tDiv = 10, pDiv = 10;

            for (int t = 0; t <= tDiv; t++)
            {
                double theta = t * Math.PI / tDiv;
                for (int p = 0; p <= pDiv; p++)
                {
                    double phi = p * 2 * Math.PI / pDiv;
                    double x = center.X + size * Math.Sin(theta) * Math.Cos(phi);
                    double y = center.Y + size * Math.Sin(theta) * Math.Sin(phi);
                    double z = center.Z + size * Math.Cos(theta);
                    mesh.Positions.Add(new Point3D(x, y, z)); 
                }
            }

            for (int t = 0; t < tDiv; t++)
            {
                for (int p = 0; p < pDiv; p++)
                {
                    int a = t * (pDiv + 1) + p;
                    int b = a + pDiv + 1;
                    int c = a + 1;
                    int d = b + 1;
                    mesh.TriangleIndices.Add(a);
                    mesh.TriangleIndices.Add(b);
                    mesh.TriangleIndices.Add(c);
                    mesh.TriangleIndices.Add(c);
                    mesh.TriangleIndices.Add(b);
                    mesh.TriangleIndices.Add(d);
                }
            }

            Material material = new DiffuseMaterial(new SolidColorBrush(color));
            return new GeometryModel3D(mesh, material);
        }
    }
}
