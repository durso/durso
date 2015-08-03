<?php

/**
 * Description of intext
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class intext extends paired{
    protected $text;
    
    public function __construct($tag,$value) {
        parent::__construct();
        $this->tag = $tag;
        if($value){
            $this->setText($value);
        }
    }
    public function changeText($value){
        assert(!is_null($this->text));
        $this->text->setValue($value);
    }
    public function setText($value){
        assert(is_null($this->text));
        $this->text = new text($value);
        $this->addComponent($this->text);
    }

}