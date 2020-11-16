#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"

#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif

using namespace binpack;
using namespace chess;

struct HalfKP {

    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;
    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static Square orient(Color color, Square sq)
    {
        if (color == Color::White)
        {
            return sq;
        }
        else
        {
            // IMPORTANT: for now we use rotate180 instead of rank flip
            //            for compatibility with the stockfish master branch.
            //            Note that this is inconsistent with nodchip/master.
            return sq.flippedVertically().flippedHorizontally();
        }
    }

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, int& counter, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            int idx = counter * 2;
            counter += 1;
            features[idx] = i;
            features[idx + 1] = feature_index(color, orient(color, ksq), sq, p);
        }
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, int& counter, Color color)
    {
        T::fill_features_sparse(i, e, features, counter, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];

        num_active_white_features = 0;
        num_active_black_features = 0;

        std::memset(white, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));
        std::memset(black, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int* white;
    int* black;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        FeatureSet<Ts...>::fill_features_sparse(i, e, white, num_active_white_features, Color::White);
        FeatureSet<Ts...>::fill_features_sparse(i, e, black, num_active_black_features, Color::Black);
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(const char* filename, bool cyclic) :
        m_stream(training_data::open_sfen_input_file(filename, cyclic))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(const char* filename, bool cyclic) :
        BaseType(filename, cyclic)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : AsyncStream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = AsyncStream<StorageT>;

    FeaturedBatchStream(const char* filename, int batch_size, bool cyclic) :
        BaseType(filename, cyclic),
        m_batch_size(batch_size)
    {

    }

    StorageT* next() override
    {
        for(;;)
        {
            auto cur = std::move(BaseType::m_next);
            if (cur.valid())
            {
                // we have to wait for this to complete before scheduling the next one
                cur.wait();
            }

            BaseType::m_next = std::async(std::launch::async, [this]() {
                std::vector<TrainingDataEntry> entries;
                entries.reserve(m_batch_size);

                for(int i = 0; i < m_batch_size; ++i)
                {
                    auto value = BaseType::m_stream->next();
                    if (value.has_value())
                    {
                        entries.emplace_back(*value);
                    }
                    else
                    {
                        break;
                    }
                }

                return new StorageT(FeatureSet{}, entries);
            });

            if (cur.valid())
            {
                return cur.get();
            }
        }
    }

private:
    int m_batch_size;
};

template <template <typename, typename> typename StreamT, typename StorageT, typename... ArgsTs>
StreamT<FeatureSet<HalfKP>, StorageT>* create_stream(std::string feature_set, ArgsTs&&... args)
{
    if (feature_set == "HalfKP")
    {
        return new StreamT<FeatureSet<HalfKP>, StorageT>(std::forward<ArgsTs>(args)...);
    }
    else
    {
        return nullptr;
    }
}

extern "C" {

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set, const char* filename, int batch_size, bool cyclic)
    {
        return create_stream<FeaturedBatchStream, SparseBatch>(feature_set, filename, batch_size, cyclic);
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}
