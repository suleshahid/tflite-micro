// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: monster_test
#ifndef GRPC_monster_5ftest__INCLUDED
#define GRPC_monster_5ftest__INCLUDED

#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

#include "flatbuffers/grpc.h"
#include "monster_test_generated.h"

namespace grpc {
class CompletionQueue;
class Channel;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace MyGame {
namespace Example {

class MonsterStorage final {
 public:
  static constexpr char const *service_full_name() {
    return "MyGame.Example.MonsterStorage";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Store(
        ::grpc::ClientContext *context,
        const flatbuffers::grpc::Message<Monster> &request,
        flatbuffers::grpc::Message<Stat> *response) = 0;
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        flatbuffers::grpc::Message<Stat>>>
    AsyncStore(::grpc::ClientContext *context,
               const flatbuffers::grpc::Message<Monster> &request,
               ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          flatbuffers::grpc::Message<Stat>>>(
          AsyncStoreRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        flatbuffers::grpc::Message<Stat>>>
    PrepareAsyncStore(::grpc::ClientContext *context,
                      const flatbuffers::grpc::Message<Monster> &request,
                      ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          flatbuffers::grpc::Message<Stat>>>(
          PrepareAsyncStoreRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientReaderInterface<flatbuffers::grpc::Message<Monster>>>
    Retrieve(::grpc::ClientContext *context,
             const flatbuffers::grpc::Message<Stat> &request) {
      return std::unique_ptr<
          ::grpc::ClientReaderInterface<flatbuffers::grpc::Message<Monster>>>(
          RetrieveRaw(context, request));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncReaderInterface<flatbuffers::grpc::Message<Monster>>>
    AsyncRetrieve(::grpc::ClientContext *context,
                  const flatbuffers::grpc::Message<Stat> &request,
                  ::grpc::CompletionQueue *cq, void *tag) {
      return std::unique_ptr<::grpc::ClientAsyncReaderInterface<
          flatbuffers::grpc::Message<Monster>>>(
          AsyncRetrieveRaw(context, request, cq, tag));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncReaderInterface<flatbuffers::grpc::Message<Monster>>>
    PrepareAsyncRetrieve(::grpc::ClientContext *context,
                         const flatbuffers::grpc::Message<Stat> &request,
                         ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<::grpc::ClientAsyncReaderInterface<
          flatbuffers::grpc::Message<Monster>>>(
          PrepareAsyncRetrieveRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientWriterInterface<flatbuffers::grpc::Message<Monster>>>
    GetMaxHitPoint(::grpc::ClientContext *context,
                   flatbuffers::grpc::Message<Stat> *response) {
      return std::unique_ptr<
          ::grpc::ClientWriterInterface<flatbuffers::grpc::Message<Monster>>>(
          GetMaxHitPointRaw(context, response));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncWriterInterface<flatbuffers::grpc::Message<Monster>>>
    AsyncGetMaxHitPoint(::grpc::ClientContext *context,
                        flatbuffers::grpc::Message<Stat> *response,
                        ::grpc::CompletionQueue *cq, void *tag) {
      return std::unique_ptr<::grpc::ClientAsyncWriterInterface<
          flatbuffers::grpc::Message<Monster>>>(
          AsyncGetMaxHitPointRaw(context, response, cq, tag));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncWriterInterface<flatbuffers::grpc::Message<Monster>>>
    PrepareAsyncGetMaxHitPoint(::grpc::ClientContext *context,
                               flatbuffers::grpc::Message<Stat> *response,
                               ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<::grpc::ClientAsyncWriterInterface<
          flatbuffers::grpc::Message<Monster>>>(
          PrepareAsyncGetMaxHitPointRaw(context, response, cq));
    }
    std::unique_ptr<::grpc::ClientReaderWriterInterface<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>>
    GetMinMaxHitPoints(::grpc::ClientContext *context) {
      return std::unique_ptr<::grpc::ClientReaderWriterInterface<
          flatbuffers::grpc::Message<Monster>,
          flatbuffers::grpc::Message<Stat>>>(GetMinMaxHitPointsRaw(context));
    }
    std::unique_ptr<::grpc::ClientAsyncReaderWriterInterface<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>>
    AsyncGetMinMaxHitPoints(::grpc::ClientContext *context,
                            ::grpc::CompletionQueue *cq, void *tag) {
      return std::unique_ptr<::grpc::ClientAsyncReaderWriterInterface<
          flatbuffers::grpc::Message<Monster>,
          flatbuffers::grpc::Message<Stat>>>(
          AsyncGetMinMaxHitPointsRaw(context, cq, tag));
    }
    std::unique_ptr<::grpc::ClientAsyncReaderWriterInterface<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>>
    PrepareAsyncGetMinMaxHitPoints(::grpc::ClientContext *context,
                                   ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<::grpc::ClientAsyncReaderWriterInterface<
          flatbuffers::grpc::Message<Monster>,
          flatbuffers::grpc::Message<Stat>>>(
          PrepareAsyncGetMinMaxHitPointsRaw(context, cq));
    }

   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        flatbuffers::grpc::Message<Stat>>
        *AsyncStoreRaw(::grpc::ClientContext *context,
                       const flatbuffers::grpc::Message<Monster> &request,
                       ::grpc::CompletionQueue *cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        flatbuffers::grpc::Message<Stat>> *
    PrepareAsyncStoreRaw(::grpc::ClientContext *context,
                         const flatbuffers::grpc::Message<Monster> &request,
                         ::grpc::CompletionQueue *cq) = 0;
    virtual ::grpc::ClientReaderInterface<flatbuffers::grpc::Message<Monster>>
        *RetrieveRaw(::grpc::ClientContext *context,
                     const flatbuffers::grpc::Message<Stat> &request) = 0;
    virtual ::grpc::ClientAsyncReaderInterface<
        flatbuffers::grpc::Message<Monster>>
        *AsyncRetrieveRaw(::grpc::ClientContext *context,
                          const flatbuffers::grpc::Message<Stat> &request,
                          ::grpc::CompletionQueue *cq, void *tag) = 0;
    virtual ::grpc::ClientAsyncReaderInterface<
        flatbuffers::grpc::Message<Monster>> *
    PrepareAsyncRetrieveRaw(::grpc::ClientContext *context,
                            const flatbuffers::grpc::Message<Stat> &request,
                            ::grpc::CompletionQueue *cq) = 0;
    virtual ::grpc::ClientWriterInterface<flatbuffers::grpc::Message<Monster>>
        *GetMaxHitPointRaw(::grpc::ClientContext *context,
                           flatbuffers::grpc::Message<Stat> *response) = 0;
    virtual ::grpc::ClientAsyncWriterInterface<
        flatbuffers::grpc::Message<Monster>>
        *AsyncGetMaxHitPointRaw(::grpc::ClientContext *context,
                                flatbuffers::grpc::Message<Stat> *response,
                                ::grpc::CompletionQueue *cq, void *tag) = 0;
    virtual ::grpc::ClientAsyncWriterInterface<
        flatbuffers::grpc::Message<Monster>> *
    PrepareAsyncGetMaxHitPointRaw(::grpc::ClientContext *context,
                                  flatbuffers::grpc::Message<Stat> *response,
                                  ::grpc::CompletionQueue *cq) = 0;
    virtual ::grpc::ClientReaderWriterInterface<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>
        *GetMinMaxHitPointsRaw(::grpc::ClientContext *context) = 0;
    virtual ::grpc::ClientAsyncReaderWriterInterface<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>
        *AsyncGetMinMaxHitPointsRaw(::grpc::ClientContext *context,
                                    ::grpc::CompletionQueue *cq, void *tag) = 0;
    virtual ::grpc::ClientAsyncReaderWriterInterface<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>
        *PrepareAsyncGetMinMaxHitPointsRaw(::grpc::ClientContext *context,
                                           ::grpc::CompletionQueue *cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface> &channel);
    ::grpc::Status Store(::grpc::ClientContext *context,
                         const flatbuffers::grpc::Message<Monster> &request,
                         flatbuffers::grpc::Message<Stat> *response) override;
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<Stat>>>
    AsyncStore(::grpc::ClientContext *context,
               const flatbuffers::grpc::Message<Monster> &request,
               ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<Stat>>>(
          AsyncStoreRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<Stat>>>
    PrepareAsyncStore(::grpc::ClientContext *context,
                      const flatbuffers::grpc::Message<Monster> &request,
                      ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<Stat>>>(
          PrepareAsyncStoreRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientReader<flatbuffers::grpc::Message<Monster>>>
    Retrieve(::grpc::ClientContext *context,
             const flatbuffers::grpc::Message<Stat> &request) {
      return std::unique_ptr<
          ::grpc::ClientReader<flatbuffers::grpc::Message<Monster>>>(
          RetrieveRaw(context, request));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncReader<flatbuffers::grpc::Message<Monster>>>
    AsyncRetrieve(::grpc::ClientContext *context,
                  const flatbuffers::grpc::Message<Stat> &request,
                  ::grpc::CompletionQueue *cq, void *tag) {
      return std::unique_ptr<
          ::grpc::ClientAsyncReader<flatbuffers::grpc::Message<Monster>>>(
          AsyncRetrieveRaw(context, request, cq, tag));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncReader<flatbuffers::grpc::Message<Monster>>>
    PrepareAsyncRetrieve(::grpc::ClientContext *context,
                         const flatbuffers::grpc::Message<Stat> &request,
                         ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncReader<flatbuffers::grpc::Message<Monster>>>(
          PrepareAsyncRetrieveRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientWriter<flatbuffers::grpc::Message<Monster>>>
    GetMaxHitPoint(::grpc::ClientContext *context,
                   flatbuffers::grpc::Message<Stat> *response) {
      return std::unique_ptr<
          ::grpc::ClientWriter<flatbuffers::grpc::Message<Monster>>>(
          GetMaxHitPointRaw(context, response));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncWriter<flatbuffers::grpc::Message<Monster>>>
    AsyncGetMaxHitPoint(::grpc::ClientContext *context,
                        flatbuffers::grpc::Message<Stat> *response,
                        ::grpc::CompletionQueue *cq, void *tag) {
      return std::unique_ptr<
          ::grpc::ClientAsyncWriter<flatbuffers::grpc::Message<Monster>>>(
          AsyncGetMaxHitPointRaw(context, response, cq, tag));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncWriter<flatbuffers::grpc::Message<Monster>>>
    PrepareAsyncGetMaxHitPoint(::grpc::ClientContext *context,
                               flatbuffers::grpc::Message<Stat> *response,
                               ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncWriter<flatbuffers::grpc::Message<Monster>>>(
          PrepareAsyncGetMaxHitPointRaw(context, response, cq));
    }
    std::unique_ptr<::grpc::ClientReaderWriter<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>>
    GetMinMaxHitPoints(::grpc::ClientContext *context) {
      return std::unique_ptr<
          ::grpc::ClientReaderWriter<flatbuffers::grpc::Message<Monster>,
                                     flatbuffers::grpc::Message<Stat>>>(
          GetMinMaxHitPointsRaw(context));
    }
    std::unique_ptr<::grpc::ClientAsyncReaderWriter<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>>
    AsyncGetMinMaxHitPoints(::grpc::ClientContext *context,
                            ::grpc::CompletionQueue *cq, void *tag) {
      return std::unique_ptr<
          ::grpc::ClientAsyncReaderWriter<flatbuffers::grpc::Message<Monster>,
                                          flatbuffers::grpc::Message<Stat>>>(
          AsyncGetMinMaxHitPointsRaw(context, cq, tag));
    }
    std::unique_ptr<::grpc::ClientAsyncReaderWriter<
        flatbuffers::grpc::Message<Monster>, flatbuffers::grpc::Message<Stat>>>
    PrepareAsyncGetMinMaxHitPoints(::grpc::ClientContext *context,
                                   ::grpc::CompletionQueue *cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncReaderWriter<flatbuffers::grpc::Message<Monster>,
                                          flatbuffers::grpc::Message<Stat>>>(
          PrepareAsyncGetMinMaxHitPointsRaw(context, cq));
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<Stat>>
        *AsyncStoreRaw(::grpc::ClientContext *context,
                       const flatbuffers::grpc::Message<Monster> &request,
                       ::grpc::CompletionQueue *cq) override;
    ::grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<Stat>> *
    PrepareAsyncStoreRaw(::grpc::ClientContext *context,
                         const flatbuffers::grpc::Message<Monster> &request,
                         ::grpc::CompletionQueue *cq) override;
    ::grpc::ClientReader<flatbuffers::grpc::Message<Monster>> *RetrieveRaw(
        ::grpc::ClientContext *context,
        const flatbuffers::grpc::Message<Stat> &request) override;
    ::grpc::ClientAsyncReader<flatbuffers::grpc::Message<Monster>>
        *AsyncRetrieveRaw(::grpc::ClientContext *context,
                          const flatbuffers::grpc::Message<Stat> &request,
                          ::grpc::CompletionQueue *cq, void *tag) override;
    ::grpc::ClientAsyncReader<flatbuffers::grpc::Message<Monster>> *
    PrepareAsyncRetrieveRaw(::grpc::ClientContext *context,
                            const flatbuffers::grpc::Message<Stat> &request,
                            ::grpc::CompletionQueue *cq) override;
    ::grpc::ClientWriter<flatbuffers::grpc::Message<Monster>>
        *GetMaxHitPointRaw(::grpc::ClientContext *context,
                           flatbuffers::grpc::Message<Stat> *response) override;
    ::grpc::ClientAsyncWriter<flatbuffers::grpc::Message<Monster>> *
    AsyncGetMaxHitPointRaw(::grpc::ClientContext *context,
                           flatbuffers::grpc::Message<Stat> *response,
                           ::grpc::CompletionQueue *cq, void *tag) override;
    ::grpc::ClientAsyncWriter<flatbuffers::grpc::Message<Monster>> *
    PrepareAsyncGetMaxHitPointRaw(::grpc::ClientContext *context,
                                  flatbuffers::grpc::Message<Stat> *response,
                                  ::grpc::CompletionQueue *cq) override;
    ::grpc::ClientReaderWriter<flatbuffers::grpc::Message<Monster>,
                               flatbuffers::grpc::Message<Stat>>
        *GetMinMaxHitPointsRaw(::grpc::ClientContext *context) override;
    ::grpc::ClientAsyncReaderWriter<flatbuffers::grpc::Message<Monster>,
                                    flatbuffers::grpc::Message<Stat>> *
    AsyncGetMinMaxHitPointsRaw(::grpc::ClientContext *context,
                               ::grpc::CompletionQueue *cq, void *tag) override;
    ::grpc::ClientAsyncReaderWriter<flatbuffers::grpc::Message<Monster>,
                                    flatbuffers::grpc::Message<Stat>> *
    PrepareAsyncGetMinMaxHitPointsRaw(::grpc::ClientContext *context,
                                      ::grpc::CompletionQueue *cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_Store_;
    const ::grpc::internal::RpcMethod rpcmethod_Retrieve_;
    const ::grpc::internal::RpcMethod rpcmethod_GetMaxHitPoint_;
    const ::grpc::internal::RpcMethod rpcmethod_GetMinMaxHitPoints_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface> &channel,
      const ::grpc::StubOptions &options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Store(
        ::grpc::ServerContext *context,
        const flatbuffers::grpc::Message<Monster> *request,
        flatbuffers::grpc::Message<Stat> *response);
    virtual ::grpc::Status Retrieve(
        ::grpc::ServerContext *context,
        const flatbuffers::grpc::Message<Stat> *request,
        ::grpc::ServerWriter<flatbuffers::grpc::Message<Monster>> *writer);
    virtual ::grpc::Status GetMaxHitPoint(
        ::grpc::ServerContext *context,
        ::grpc::ServerReader<flatbuffers::grpc::Message<Monster>> *reader,
        flatbuffers::grpc::Message<Stat> *response);
    virtual ::grpc::Status GetMinMaxHitPoints(
        ::grpc::ServerContext *context,
        ::grpc::ServerReaderWriter<flatbuffers::grpc::Message<Stat>,
                                   flatbuffers::grpc::Message<Monster>>
            *stream);
  };
  template<class BaseClass> class WithAsyncMethod_Store : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithAsyncMethod_Store() { ::grpc::Service::MarkMethodAsync(0); }
    ~WithAsyncMethod_Store() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Store(
        ::grpc::ServerContext * /*context*/,
        const flatbuffers::grpc::Message<Monster> * /*request*/,
        flatbuffers::grpc::Message<Stat> * /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestStore(
        ::grpc::ServerContext *context,
        flatbuffers::grpc::Message<Monster> *request,
        ::grpc::ServerAsyncResponseWriter<flatbuffers::grpc::Message<Stat>>
            *response,
        ::grpc::CompletionQueue *new_call_cq,
        ::grpc::ServerCompletionQueue *notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
  template<class BaseClass> class WithAsyncMethod_Retrieve : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithAsyncMethod_Retrieve() { ::grpc::Service::MarkMethodAsync(1); }
    ~WithAsyncMethod_Retrieve() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Retrieve(
        ::grpc::ServerContext * /*context*/,
        const flatbuffers::grpc::Message<Stat> * /*request*/,
        ::grpc::ServerWriter<flatbuffers::grpc::Message<Monster>> * /*writer*/)
        final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestRetrieve(
        ::grpc::ServerContext *context,
        flatbuffers::grpc::Message<Stat> *request,
        ::grpc::ServerAsyncWriter<flatbuffers::grpc::Message<Monster>> *writer,
        ::grpc::CompletionQueue *new_call_cq,
        ::grpc::ServerCompletionQueue *notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncServerStreaming(
          1, context, request, writer, new_call_cq, notification_cq, tag);
    }
  };
  template<class BaseClass>
  class WithAsyncMethod_GetMaxHitPoint : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithAsyncMethod_GetMaxHitPoint() { ::grpc::Service::MarkMethodAsync(2); }
    ~WithAsyncMethod_GetMaxHitPoint() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetMaxHitPoint(
        ::grpc::ServerContext * /*context*/,
        ::grpc::ServerReader<flatbuffers::grpc::Message<Monster>> * /*reader*/,
        flatbuffers::grpc::Message<Stat> *response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetMaxHitPoint(
        ::grpc::ServerContext *context,
        ::grpc::ServerAsyncReader<flatbuffers::grpc::Message<Stat>,
                                  flatbuffers::grpc::Message<Monster>> *reader,
        ::grpc::CompletionQueue *new_call_cq,
        ::grpc::ServerCompletionQueue *notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncClientStreaming(
          2, context, reader, new_call_cq, notification_cq, tag);
    }
  };
  template<class BaseClass>
  class WithAsyncMethod_GetMinMaxHitPoints : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithAsyncMethod_GetMinMaxHitPoints() {
      ::grpc::Service::MarkMethodAsync(3);
    }
    ~WithAsyncMethod_GetMinMaxHitPoints() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetMinMaxHitPoints(
        ::grpc::ServerContext * /*context*/,
        ::grpc::ServerReaderWriter<flatbuffers::grpc::Message<Stat>,
                                   flatbuffers::grpc::Message<Monster>>
            * /*stream*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetMinMaxHitPoints(
        ::grpc::ServerContext *context,
        ::grpc::ServerAsyncReaderWriter<flatbuffers::grpc::Message<Stat>,
                                        flatbuffers::grpc::Message<Monster>>
            *stream,
        ::grpc::CompletionQueue *new_call_cq,
        ::grpc::ServerCompletionQueue *notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncBidiStreaming(
          3, context, stream, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Store<
      WithAsyncMethod_Retrieve<WithAsyncMethod_GetMaxHitPoint<
          WithAsyncMethod_GetMinMaxHitPoints<Service>>>>
      AsyncService;
  template<class BaseClass> class WithGenericMethod_Store : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithGenericMethod_Store() { ::grpc::Service::MarkMethodGeneric(0); }
    ~WithGenericMethod_Store() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Store(
        ::grpc::ServerContext * /*context*/,
        const flatbuffers::grpc::Message<Monster> * /*request*/,
        flatbuffers::grpc::Message<Stat> * /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template<class BaseClass>
  class WithGenericMethod_Retrieve : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithGenericMethod_Retrieve() { ::grpc::Service::MarkMethodGeneric(1); }
    ~WithGenericMethod_Retrieve() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Retrieve(
        ::grpc::ServerContext * /*context*/,
        const flatbuffers::grpc::Message<Stat> * /*request*/,
        ::grpc::ServerWriter<flatbuffers::grpc::Message<Monster>> * /*writer*/)
        final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template<class BaseClass>
  class WithGenericMethod_GetMaxHitPoint : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithGenericMethod_GetMaxHitPoint() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_GetMaxHitPoint() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetMaxHitPoint(
        ::grpc::ServerContext * /*context*/,
        ::grpc::ServerReader<flatbuffers::grpc::Message<Monster>> * /*reader*/,
        flatbuffers::grpc::Message<Stat> * /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template<class BaseClass>
  class WithGenericMethod_GetMinMaxHitPoints : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithGenericMethod_GetMinMaxHitPoints() {
      ::grpc::Service::MarkMethodGeneric(3);
    }
    ~WithGenericMethod_GetMinMaxHitPoints() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetMinMaxHitPoints(
        ::grpc::ServerContext * /*context*/,
        ::grpc::ServerReaderWriter<flatbuffers::grpc::Message<Stat>,
                                   flatbuffers::grpc::Message<Monster>>
            * /*stream*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template<class BaseClass>
  class WithStreamedUnaryMethod_Store : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithStreamedUnaryMethod_Store() {
      ::grpc::Service::MarkMethodStreamed(
          0, new ::grpc::internal::StreamedUnaryHandler<
                 flatbuffers::grpc::Message<Monster>,
                 flatbuffers::grpc::Message<Stat>>(std::bind(
                 &WithStreamedUnaryMethod_Store<BaseClass>::StreamedStore, this,
                 std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Store() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Store(
        ::grpc::ServerContext * /*context*/,
        const flatbuffers::grpc::Message<Monster> * /*request*/,
        flatbuffers::grpc::Message<Stat> * /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedStore(
        ::grpc::ServerContext *context,
        ::grpc::ServerUnaryStreamer<flatbuffers::grpc::Message<Monster>,
                                    flatbuffers::grpc::Message<Stat>>
            *server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_Store<Service> StreamedUnaryService;
  template<class BaseClass>
  class WithSplitStreamingMethod_Retrieve : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service * /*service*/) {}

   public:
    WithSplitStreamingMethod_Retrieve() {
      ::grpc::Service::MarkMethodStreamed(
          1,
          new ::grpc::internal::SplitServerStreamingHandler<
              flatbuffers::grpc::Message<Stat>,
              flatbuffers::grpc::Message<Monster>>(std::bind(
              &WithSplitStreamingMethod_Retrieve<BaseClass>::StreamedRetrieve,
              this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithSplitStreamingMethod_Retrieve() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Retrieve(
        ::grpc::ServerContext * /*context*/,
        const flatbuffers::grpc::Message<Stat> * /*request*/,
        ::grpc::ServerWriter<flatbuffers::grpc::Message<Monster>> * /*writer*/)
        final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with split streamed
    virtual ::grpc::Status StreamedRetrieve(
        ::grpc::ServerContext *context,
        ::grpc::ServerSplitStreamer<flatbuffers::grpc::Message<Stat>,
                                    flatbuffers::grpc::Message<Monster>>
            *server_split_streamer) = 0;
  };
  typedef WithSplitStreamingMethod_Retrieve<Service> SplitStreamedService;
  typedef WithStreamedUnaryMethod_Store<
      WithSplitStreamingMethod_Retrieve<Service>>
      StreamedService;
};

}  // namespace Example
}  // namespace MyGame

#endif  // GRPC_monster_5ftest__INCLUDED
