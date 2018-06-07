using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SAM.Core.DataProcessing;

namespace SamDSPbyGPU
{
    class Program
    {
        private const int NUMBER_RECORDS = 1400;
        private const int RECORD_LENGTH = 150;
        private static float[,] _bScan;
        private static readonly string dataPath = @"D:\CurrentWork\Data\29e79\e79\29e79_1.dat";

        static void Main(string[] args)
        {
            Console.WriteLine("Loading sample data...");
            _bScan = SAM.Core.DataProcessing.SamBScanLoader.LoadFloat(dataPath, NUMBER_RECORDS, RECORD_LENGTH);
            Console.WriteLine("Done!");

            //var aScan = new float[RECORD_LENGTH];
            //for (int i = 0; i <= RECORD_LENGTH; i++)
            //    aScan[i] = _bScan[0, i];

            Action<Action> measure = (body) =>
            {
                var startTime = DateTime.Now;
                body();
                Console.WriteLine("Time Elapsed: {0} ", (DateTime.Now - startTime).Milliseconds);
            };

            //var random = new Random();
            //var aScan = new double[100000];
            //for (int i = 0; i < 100000; i++)
            //{
            //    double rand = ((double)random.Next(2000)) / 2000;
            //    rand = rand * 2 + -1;
            //    aScan[i] = rand;
            //}

            //measure(() => DIP.FHT(aScan, FourierTransform.Direction.Forward));

            var random = new Random();
            var aScan = new double[500];
            for (int j = 0; j < 500; j++)
            {
                double rand = ((double)random.Next(2000)) / 2000;
                rand = rand * 2 + -1;
                aScan[j] = rand;
            }

            measure(() =>
            {
                for (int i = 0; i < 2000; i++)
                {
                    //var aScan = new double[RECORD_LENGTH];
                    //for (int j = 0; j < RECORD_LENGTH; j++)
                    //    aScan[j] = _bScan[0, j];


                    DIP.FHT(aScan, FourierTransform.Direction.Forward);
                }
            });

            //var tasks = new Task[NUMBER_RECORDS];
            //for (int i = 0; i < NUMBER_RECORDS; i++)
            //{
            //    var aScan = new double[RECORD_LENGTH];
            //    for (int j = 0; j < RECORD_LENGTH; j++)
            //        aScan[j] = _bScan[0, j];
            //    tasks[i] = Task.Factory.StartNew(() => DIP.FHT(aScan, FourierTransform.Direction.Forward));
            //}

            //measure(() => Task.WaitAll(tasks));
            Console.ReadKey();
        }
    }
}
