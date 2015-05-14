<?php

namespace app\controller;
use app\controller\base\controller as baseController, cache\cache, app\request;
use library\utils;
use library\layout\elements\element;
use library\layout\elements\button;
use library\layout\elements\form;
use library\layout\elements\group;
use library\layout\elements\input;
use library\event\event;
use library\layout\elements\script;

class index extends baseController{

    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }

    public function index(){
        /*
         * Cache version:
        $cache = new cache($this->controller,$this->action);
        if(!$cache->init()){
            $this->addView();
            $this->view->files($this->render);
            $this->view->render();
            try{
                $cache->run();
            }  catch (\Exception $e){
                utils::log($e->getMessage());
            }
        }
        $cache->getCache();
       */
        $form = new form("index.php");
        $group = new group();
        $input = new input("text","Hello world","Escreva:");
        $group2 = new group();
        $button = new button("Click me","button");
        $button->bind('click', function($button){
            $button->setScript(true);
            $button->addClassName("red");
            $button->changeValue("Clicked");
            $group2 = $button->getParent();
            $group2->setScript(true);
            $button2 = new button('Botao2','button');
            $group2->addChild($button2,$group2->getId(),"prepend");
        });
        $group->addChild($input);
        $group2->addChild($button);
        $form->addChild($group);
        $form->addChild($group2);
            
        if(request::isAjax()){
            event::trigger($button->getId(), $this->query["event"]);
            echo script::getResponse();
            exit;
        } else {
            $this->html($form);
        }
    }
    
    
    

}